"""Unit tests for heatmap.py. Run with: python test_heatmap.py"""
import sys
sys.path.insert(0, '.')

import numpy as np
from heatmap import (
    ROOM_W, ROOM_H, GRID_W, GRID_H, CELL_W, CELL_H,
    xy_to_cell, cell_to_xy,
    xy_to_heatmap, heatmap_to_xy,
    count_target, count_target_onehot,
    make_frame_stack, frame_stack_view,
)


def test_coord_round_trip():
    """xy_to_cell -> cell_to_xy at integer cell centres should be identity."""
    xs = np.array([0.1, 1.0, 2.4, 3.5, 4.7])
    ys = np.array([0.1, 2.0, 3.6, 5.0, 7.1])
    col, row = xy_to_cell(xs, ys)
    x_back, y_back = cell_to_xy(col, row)
    np.testing.assert_allclose(x_back, xs, atol=1e-5)
    np.testing.assert_allclose(y_back, ys, atol=1e-5)
    print("PASS coord_round_trip")


def test_grid_corners():
    """Corner positions should map to the corner cells."""
    # Bottom-left corner of room ~ cell (0, 0) (col=0, row=0)
    col, row = xy_to_cell(np.array([0.0]), np.array([0.0]))
    # Cell *centre* at j=0 is x = 0.5*CELL_W = 0.1, so x=0 maps near col=-0.5 ~ clipped
    # Top-right corner ~ cell (GRID_W-1, GRID_H-1)
    col, row = xy_to_cell(np.array([ROOM_W]), np.array([ROOM_H]))
    assert int(np.floor(col[0])) == GRID_W - 1, f"got col {col}"
    assert int(np.floor(row[0])) == GRID_H - 1, f"got row {row}"
    print("PASS grid_corners")


def test_heatmap_empty():
    """Empty mask -> all-zeros heatmap."""
    xy = np.zeros((4, 2))
    mask = np.zeros(4, dtype=bool)
    h = xy_to_heatmap(xy, mask)
    assert h.shape == (GRID_H, GRID_W)
    assert h.sum() == 0.0
    print("PASS heatmap_empty")


def test_heatmap_peak_at_position():
    """Single person -> peak ~ 1.0 at their cell."""
    x, y = 2.4, 3.6   # centre of room
    xy = np.array([[x, y], [0, 0], [0, 0], [0, 0]])
    mask = np.array([True, False, False, False])
    h = xy_to_heatmap(xy, mask)
    assert h.shape == (GRID_H, GRID_W)
    peak = h.max()
    assert 0.99 <= peak <= 1.001, f"peak should be ~1, got {peak}"

    # Peak location should match cell index
    col_expected, row_expected = xy_to_cell(np.array([x]), np.array([y]))
    peak_idx = np.unravel_index(np.argmax(h), h.shape)
    # Peak row, col should be at most 1 cell off (sub-pixel rounding)
    assert abs(peak_idx[0] - row_expected[0]) < 1.5, f"row mismatch: peak={peak_idx[0]}, expected~{row_expected[0]}"
    assert abs(peak_idx[1] - col_expected[0]) < 1.5, f"col mismatch: peak={peak_idx[1]}, expected~{col_expected[0]}"
    print("PASS heatmap_peak_at_position")


def test_heatmap_multi_person():
    """3 people -> 3 distinct peaks, each near 1.0."""
    xy = np.array([
        [1.0, 2.0],
        [3.5, 5.0],
        [4.0, 1.0],
        [0.0, 0.0],   # masked off
    ])
    mask = np.array([True, True, True, False])
    h = xy_to_heatmap(xy, mask)
    assert 0.99 <= h.max() <= 1.001
    # Sum should be roughly 3 * (gaussian volume) but we don't pin that;
    # we just check 3 detectable peaks via the extractor.
    extracted = heatmap_to_xy(h, k=3)
    assert extracted.shape == (3, 2)

    # Each true person should be matched by an extracted peak within sqrt(2)*CELL diagonal.
    matched = np.zeros(3, dtype=bool)
    for true_xy in xy[mask]:
        dists = np.linalg.norm(extracted - true_xy, axis=1)
        j = int(np.argmin(dists))
        if dists[j] < 0.30:    # 30 cm tolerance: 1.5 cells
            matched[j] = True
    assert matched.sum() == 3, f"only matched {matched.sum()}/3 people"
    print("PASS heatmap_multi_person")


def test_heatmap_to_xy_round_trip():
    """xy -> heatmap -> xy within sub-cell tolerance."""
    xy_in = np.array([
        [1.5, 3.0],
        [2.8, 5.5],
        [0.0, 0.0],
        [0.0, 0.0],
    ])
    mask = np.array([True, True, False, False])
    h = xy_to_heatmap(xy_in, mask)
    xy_out = heatmap_to_xy(h, k=2)
    assert xy_out.shape == (2, 2)
    # Match outputs to inputs greedily
    used = set()
    errs = []
    for true_xy in xy_in[mask]:
        dists = np.linalg.norm(xy_out - true_xy, axis=1)
        for j in np.argsort(dists):
            if int(j) not in used:
                used.add(int(j))
                errs.append(dists[j])
                break
    print(f"  round-trip per-person error (m): {[f'{e:.3f}' for e in errs]}")
    # Half-cell diagonal = sqrt(2)*0.5*0.2 ~ 0.14 m. Snapped targets
    # cannot do better; sub-pixel refinement helps only on real model
    # outputs where the Gaussian is asymmetric.
    for e in errs:
        assert e < 0.15, f"round-trip error too large: {e}"
    print("PASS heatmap_to_xy_round_trip")


def test_heatmap_to_xy_k_zero():
    """k=0 -> empty array."""
    h = np.random.rand(GRID_H, GRID_W).astype(np.float32)
    xy = heatmap_to_xy(h, k=0)
    assert xy.shape == (0, 2)
    print("PASS heatmap_to_xy_k_zero")


def test_heatmap_batch():
    """Batched (T, P, 2) input."""
    T = 5
    xy = np.zeros((T, 4, 2))
    mask = np.zeros((T, 4), dtype=bool)
    xy[0, 0] = [2.4, 3.6]; mask[0, 0] = True
    xy[2, 0] = [1.0, 1.0]; mask[2, 0] = True
    xy[2, 1] = [3.5, 5.5]; mask[2, 1] = True
    h = xy_to_heatmap(xy, mask)
    assert h.shape == (T, GRID_H, GRID_W)
    assert h[1].sum() == 0       # frame 1 had no people
    assert h[3].sum() == 0       # frame 3 had no people
    assert h[0].max() > 0.99
    print("PASS heatmap_batch")


def test_count_target():
    mask = np.array([
        [False, False, False, False],   # 0
        [True,  False, False, False],   # 1
        [True,  True,  False, False],   # 2
        [True,  True,  True,  False],   # 3
        [True,  True,  True,  True],    # 4
    ])
    n = count_target(mask)
    np.testing.assert_array_equal(n, [0, 1, 2, 3, 4])

    onehot = count_target_onehot(mask)
    assert onehot.shape == (5, 5)
    np.testing.assert_array_equal(onehot.argmax(axis=1), [0, 1, 2, 3, 4])
    print("PASS count_target")


def test_make_frame_stack_basic():
    cir = np.arange(20).reshape(20, 1, 1).astype(np.float32)
    stack = make_frame_stack(cir, T_context=4)
    assert stack.shape == (20, 4, 1, 1)
    # At t=0: padding fills t=-3..-1 with cir[0]=0, so stack[0]=[0,0,0,0]
    np.testing.assert_array_equal(stack[0, :, 0, 0], [0, 0, 0, 0])
    # At t=3: should be [0, 1, 2, 3]
    np.testing.assert_array_equal(stack[3, :, 0, 0], [0, 1, 2, 3])
    # At t=10: should be [7, 8, 9, 10]
    np.testing.assert_array_equal(stack[10, :, 0, 0], [7, 8, 9, 10])
    print("PASS make_frame_stack_basic")


def test_frame_stack_view_matches():
    cir = np.random.randn(20, 6, 3, 105, 2).astype(np.float32)
    stack_full = make_frame_stack(cir, T_context=8)
    stack_view = frame_stack_view(cir, T_context=8)
    # Stride view starts at t = T_context - 1 = 7
    np.testing.assert_array_equal(stack_full[7], stack_view[0])
    np.testing.assert_array_equal(stack_full[19], stack_view[12])
    print("PASS frame_stack_view_matches")


def test_overlapping_people_clip_to_one():
    """Two people in the same cell -> peak still <= 1 (no double-counting)."""
    xy = np.array([
        [2.4, 3.6],
        [2.4, 3.6],   # exact same spot
        [0.0, 0.0], [0.0, 0.0],
    ])
    mask = np.array([True, True, False, False])
    h = xy_to_heatmap(xy, mask)
    assert h.max() <= 1.001, f"overlapping peaks should clip to 1, got {h.max()}"
    print("PASS overlapping_people_clip_to_one")


if __name__ == "__main__":
    tests = [
        test_coord_round_trip,
        test_grid_corners,
        test_heatmap_empty,
        test_heatmap_peak_at_position,
        test_heatmap_multi_person,
        test_heatmap_to_xy_round_trip,
        test_heatmap_to_xy_k_zero,
        test_heatmap_batch,
        test_count_target,
        test_make_frame_stack_basic,
        test_frame_stack_view_matches,
        test_overlapping_people_clip_to_one,
    ]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} tests passed.")
