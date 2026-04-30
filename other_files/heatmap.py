"""
heatmap.py
==========

Heatmap target generation, peak extraction, and frame stacking for the
hybrid heatmap + count-head localization model.

This module is a single source of truth shared by:
    * training notebooks  (target generation)
    * code.py             (peak extraction + frame stacking)
    * evaluation scripts  (sanity checks)

Conventions
-----------
Coordinate system: room origin at bottom-left corner.
    x in [0, ROOM_W],  y in [0, ROOM_H]   (metres)

Grid layout (matches Keras "channels last"):
    heatmap shape = (GRID_H, GRID_W, 1)
    row index     -> y axis  (row 0 is y near 0)
    col index     -> x axis  (col 0 is x near 0)

Cell centres:
    x_cell(j) = (j + 0.5) * cell_w     for j in [0, GRID_W)
    y_cell(i) = (i + 0.5) * cell_h     for i in [0, GRID_H)

Author: Sepehr (TinyML UWB project, 2026)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ============================================================
# Default room + grid constants
# ============================================================
ROOM_W = 4.8     # metres (x-axis)
ROOM_H = 7.2     # metres (y-axis)
GRID_W = 24      # cells along x
GRID_H = 36      # cells along y
CELL_W = ROOM_W / GRID_W   # 0.20 m
CELL_H = ROOM_H / GRID_H   # 0.20 m
SIGMA_CELLS = 1.5          # Gaussian half-width in cells

# Count-head categories: 0, 1, 2, 3, 4 people
MAX_PEOPLE = 4
COUNT_CLASSES = MAX_PEOPLE + 1


# ============================================================
# 1. Coordinate <-> grid helpers
# ============================================================
def xy_to_cell(x: np.ndarray, y: np.ndarray,
               grid_w: int = GRID_W, grid_h: int = GRID_H,
               room_w: float = ROOM_W, room_h: float = ROOM_H,
               ) -> tuple[np.ndarray, np.ndarray]:
    """Convert metric (x, y) to (col, row) sub-pixel cell coordinates.

    Returned values are floats; floor() gives the containing cell.
    Out-of-room positions are clipped to [0, grid-1+epsilon] so they
    still land in a valid cell.
    """
    cw, ch = room_w / grid_w, room_h / grid_h
    col = x / cw - 0.5    # so cell centre at j has col=j
    row = y / ch - 0.5
    eps = 1e-6
    col = np.clip(col, -0.5 + eps, grid_w - 0.5 - eps)
    row = np.clip(row, -0.5 + eps, grid_h - 0.5 - eps)
    return col, row


def cell_to_xy(col: np.ndarray, row: np.ndarray,
               grid_w: int = GRID_W, grid_h: int = GRID_H,
               room_w: float = ROOM_W, room_h: float = ROOM_H,
               ) -> tuple[np.ndarray, np.ndarray]:
    """Convert (col, row) -- possibly sub-pixel -- back to metric (x, y)."""
    cw, ch = room_w / grid_w, room_h / grid_h
    x = (col + 0.5) * cw
    y = (row + 0.5) * ch
    return x, y


# ============================================================
# 2. Heatmap target generator
# ============================================================
def xy_to_heatmap(people_xy: np.ndarray,
                  people_mask: np.ndarray,
                  grid_h: int = GRID_H,
                  grid_w: int = GRID_W,
                  sigma_cells: float = SIGMA_CELLS,
                  room_w: float = ROOM_W,
                  room_h: float = ROOM_H,
                  truncate: float = 4.0,
                  ) -> np.ndarray:
    """Generate ground-truth heatmap(s) from (x, y) positions.

    Parameters
    ----------
    people_xy   : (T, P, 2) float array, with P <= 4 person slots.
                  (Or shape (P, 2) for a single frame.)
    people_mask : (T, P) bool array.  (Or shape (P,) for a single frame.)
    grid_h, grid_w : output grid size.
    sigma_cells : Gaussian std-dev in cell units.
    truncate    : kernel half-width in stddevs (kernel size = 2*ceil(t*s)+1).

    Returns
    -------
    heatmap : (T, grid_h, grid_w) float32  (or (grid_h, grid_w) for one frame).

    The heatmap value at each cell is the Gaussian peak = 1.0 at each
    person's nearest cell centre; the stamp is snapped to the nearest
    cell so the target always has a clean peak == 1.0. Sub-cell
    position is recovered at inference via heatmap_to_xy(refine=True).
    Multiple overlapping people are combined via element-wise max so
    the head remains a "presence" map and BCE loss stays well-defined.
    """
    if people_xy.ndim == 2:
        # Single-frame fast path.
        out = _single_heatmap(people_xy, people_mask,
                              grid_h, grid_w, sigma_cells,
                              room_w, room_h, truncate)
        return out

    T = people_xy.shape[0]
    out = np.zeros((T, grid_h, grid_w), dtype=np.float32)
    for t in range(T):
        out[t] = _single_heatmap(people_xy[t], people_mask[t],
                                 grid_h, grid_w, sigma_cells,
                                 room_w, room_h, truncate)
    return out


def _single_heatmap(xy: np.ndarray, mask: np.ndarray,
                    grid_h: int, grid_w: int, sigma_cells: float,
                    room_w: float, room_h: float, truncate: float,
                    ) -> np.ndarray:
    out = np.zeros((grid_h, grid_w), dtype=np.float32)
    if mask.sum() == 0:
        return out

    # Stamp size in cells. Half-width in cells = ceil(truncate * sigma).
    hw = int(np.ceil(truncate * sigma_cells))

    # Pre-built unit Gaussian stamp (peak=1)
    yy, xx = np.mgrid[-hw:hw + 1, -hw:hw + 1].astype(np.float32)
    base_stamp = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma_cells ** 2))
    # Peak is exactly 1.0 by construction.

    # Apply each person.
    # We SNAP the Gaussian peak to the nearest cell, so the target
    # always has peak == 1.0 at exactly one cell. Sub-cell location is
    # recovered at inference via heatmap_to_xy() with refine=True.
    valid = np.where(mask)[0]
    for p in valid:
        col, row = xy_to_cell(xy[p, 0], xy[p, 1], grid_w, grid_h, room_w, room_h)
        col_int = int(np.round(col))
        row_int = int(np.round(row))
        col_int = max(0, min(grid_w - 1, col_int))
        row_int = max(0, min(grid_h - 1, row_int))
        stamp = base_stamp     # peak exactly 1.0 at the centre

        # Compute write window
        r0 = row_int - hw
        c0 = col_int - hw
        r1 = r0 + stamp.shape[0]
        c1 = c0 + stamp.shape[1]

        # Clip to grid
        r0c, r1c = max(0, r0), min(grid_h, r1)
        c0c, c1c = max(0, c0), min(grid_w, c1)
        if r1c <= r0c or c1c <= c0c:
            continue
        sr0, sr1 = r0c - r0, stamp.shape[0] - (r1 - r1c)
        sc0, sc1 = c0c - c0, stamp.shape[1] - (c1 - c1c)
        out[r0c:r1c, c0c:c1c] = np.maximum(
            out[r0c:r1c, c0c:c1c], stamp[sr0:sr1, sc0:sc1]
        )
    return out


# ============================================================
# 3. Peak extraction (used in code.py)
# ============================================================
def heatmap_to_xy(heatmap: np.ndarray,
                  k: int,
                  grid_h: int = GRID_H,
                  grid_w: int = GRID_W,
                  room_w: float = ROOM_W,
                  room_h: float = ROOM_H,
                  nms_radius: int = 2,
                  refine: bool = True,
                  ) -> np.ndarray:
    """Extract up to ``k`` peaks from a heatmap, return their (x, y).

    Parameters
    ----------
    heatmap   : (grid_h, grid_w) or (grid_h, grid_w, 1) float array.
    k         : number of peaks to return (the predicted person count).
    nms_radius: NMS suppression radius in cells. After picking a peak,
                a (2r+1)x(2r+1) neighbourhood around it is zeroed out
                before the next peak is picked. r=2 -> 1m at 20cm cells.
    refine    : if True, refine each peak with a 3x3 weighted-centroid
                interpolation for sub-cell accuracy.

    Returns
    -------
    xy : (k_actual, 2) float32, where k_actual <= k. Empty (0, 2) if k=0.
    """
    if heatmap.ndim == 3:
        heatmap = heatmap[..., 0]
    if k <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    h = heatmap.copy().astype(np.float32)
    out_xy = []
    for _ in range(k):
        idx = int(np.argmax(h))
        peak = h.flat[idx]
        if peak <= 0:
            break
        row, col = divmod(idx, grid_w)
        if refine:
            col_f, row_f = _refine_peak(h, row, col)
        else:
            col_f, row_f = float(col), float(row)
        x, y = cell_to_xy(np.array(col_f), np.array(row_f),
                          grid_w, grid_h, room_w, room_h)
        out_xy.append([float(x), float(y)])
        # NMS: zero the neighbourhood around the picked peak
        r0, r1 = max(0, row - nms_radius), min(grid_h, row + nms_radius + 1)
        c0, c1 = max(0, col - nms_radius), min(grid_w, col + nms_radius + 1)
        h[r0:r1, c0:c1] = 0.0
    if not out_xy:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(out_xy, dtype=np.float32)


def _refine_peak(h: np.ndarray, row: int, col: int) -> tuple[float, float]:
    """Sub-cell refinement via weighted centroid in a 3x3 window."""
    grid_h, grid_w = h.shape
    r0, r1 = max(0, row - 1), min(grid_h, row + 2)
    c0, c1 = max(0, col - 1), min(grid_w, col + 2)
    win = h[r0:r1, c0:c1]
    win = np.maximum(win, 0)
    s = win.sum()
    if s <= 0:
        return float(col), float(row)
    rr = np.arange(r0, r1)
    cc = np.arange(c0, c1)
    wr = (win.sum(axis=1) * rr).sum() / s
    wc = (win.sum(axis=0) * cc).sum() / s
    return float(wc), float(wr)


# ============================================================
# 4. Count head target / inference helper
# ============================================================
def count_target(people_mask: np.ndarray) -> np.ndarray:
    """Per-frame integer count in [0, MAX_PEOPLE].

    people_mask: (T, 4) -> (T,) int64 in {0,...,4}.
    """
    return people_mask.astype(np.int64).sum(axis=-1).clip(0, MAX_PEOPLE)


def count_target_onehot(people_mask: np.ndarray) -> np.ndarray:
    """One-hot (T, COUNT_CLASSES) target for sparse_categorical_crossentropy.
    Returned as one-hot for compatibility with categorical_crossentropy
    if needed.
    """
    n = count_target(people_mask)
    out = np.zeros((n.size, COUNT_CLASSES), dtype=np.float32)
    out[np.arange(n.size), n] = 1.0
    return out


def predict_count(softmax_probs: np.ndarray) -> int:
    """argmax of the count-head softmax."""
    return int(np.argmax(softmax_probs))


# ============================================================
# 5. Frame stacking (causal)
# ============================================================
def make_frame_stack(cir: np.ndarray,
                     T_context: int = 8,
                     ) -> np.ndarray:
    """Stack the most recent ``T_context`` frames as a new leading axis.

    Causal: stack[t] = cir[t-T_context+1 : t+1].
    Frames before t=0 are filled by replicating cir[0] (so the model
    starts with valid input even at the very first frame, accepting
    that early predictions will be poor).

    Parameters
    ----------
    cir : (T, ...) array.

    Returns
    -------
    stack : (T, T_context, ...) array, same dtype as input.

    Memory note: this materialises a (T, T_context, R, A, B, 2) tensor.
    For T=7500, T_context=8, R=6, A=3, B=105, that's
    7500*8*6*3*105*2 = 226.8 M float32 = 907 MB. That's too much.
    Train-time generators should call this on a per-batch basis or
    use a stride-trick view (see ``frame_stack_view``).
    """
    T = cir.shape[0]
    pad = np.repeat(cir[:1], T_context - 1, axis=0)
    padded = np.concatenate([pad, cir], axis=0)   # (T + T_context - 1, ...)
    out = np.empty((T, T_context) + cir.shape[1:], dtype=cir.dtype)
    for t in range(T):
        out[t] = padded[t:t + T_context]
    return out


def frame_stack_view(cir: np.ndarray, T_context: int = 8) -> np.ndarray:
    """Memory-efficient stride-trick view of make_frame_stack.

    Returns a (T, T_context, ...) view that does NOT materialise a copy.
    The view aliases the input buffer, so do not write to it. Read-only
    use in tf.data pipelines is safe.

    The first T_context-1 entries are *not* replicated; this view is
    only valid for t in [T_context-1, T). Use ``make_frame_stack`` if
    you need the early frames too.
    """
    from numpy.lib.stride_tricks import sliding_window_view
    # sliding_window_view returns shape (T - T_context + 1, ..., T_context)
    # with the window axis at the end -- we move it to position 1.
    sw = sliding_window_view(cir, T_context, axis=0)
    # sw.shape = (T - T_context + 1, ..., T_context)
    # Move the last axis (window) to position 1:
    sw = np.moveaxis(sw, -1, 1)
    return sw


__all__ = [
    "ROOM_W", "ROOM_H", "GRID_W", "GRID_H", "CELL_W", "CELL_H",
    "SIGMA_CELLS", "MAX_PEOPLE", "COUNT_CLASSES",
    "xy_to_cell", "cell_to_xy",
    "xy_to_heatmap", "heatmap_to_xy",
    "count_target", "count_target_onehot", "predict_count",
    "make_frame_stack", "frame_stack_view",
]
