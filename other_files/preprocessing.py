from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


# ============================================================
# Clutter removal
# ============================================================
class RunningMeanClutterRemover:
    """Centred running-mean subtraction along the time axis (axis 0).

    Offline-only: requires future frames. Used as the EDA baseline and
    as a sanity reference; **not** used in code.py.
    """

    def __init__(self, window: int = 50):
        if window < 2:
            raise ValueError("window must be >= 2")
        self.window = int(window)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """x: (T, ...) -> same shape, mean-subtracted along axis 0."""
        T = x.shape[0]
        W = min(self.window, T)
        kernel = np.ones(W, dtype=np.float32) / W

        # Move time axis to the end for fast convolve, then move back.
        # apply_along_axis handles arbitrary trailing shape.
        pad = W // 2
        x_padded = np.concatenate(
            [np.repeat(x[:1], pad, axis=0), x, np.repeat(x[-1:], pad, axis=0)],
            axis=0,
        )
        mean = np.apply_along_axis(
            lambda v: np.convolve(v, kernel, mode="valid"),
            axis=0,
            arr=x_padded,
        )
        # Trim if convolution length differs from T by 1 (even W edge case)
        mean = mean[:T]
        return (x - mean).astype(np.float32, copy=False)


class EMAClutterRemover:
    """Causal exponential-moving-average subtraction along axis 0.

    For each frame t:
        m[t] = (1 - alpha) * m[t-1] + alpha * x[t]
        y[t] = x[t] - m[t]

    Strictly causal -> deployable on the MCU. ``alpha`` controls the
    effective time constant: tau ~= 1/alpha frames. With alpha=0.02 and
    25 Hz sampling, tau ~= 50 frames = 2 s, matching the running-mean
    EDA window.
    """

    def __init__(self, alpha: float = 0.02, warmup: int = 50):
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = float(alpha)
        # Number of frames to skip at the start (mean is not yet stable).
        self.warmup = int(warmup)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """x: (T, ...) -> same shape, EMA-subtracted along axis 0.

        The first ``warmup`` frames receive a partial-mean correction
        (the EMA bootstraps from the first frame so it is well-defined
        from t=0; we just don't claim those frames are clean).
        """
        T = x.shape[0]
        a = self.alpha
        out = np.empty_like(x, dtype=np.float32)
        m = x[0].astype(np.float32, copy=True)
        for t in range(T):
            m = (1.0 - a) * m + a * x[t]
            out[t] = x[t] - m
        return out


def get_clutter_remover(name: str, **kw):
    name = name.lower()
    if name == "running_mean":
        return RunningMeanClutterRemover(window=kw.get("window", 50))
    if name == "ema":
        return EMAClutterRemover(
            alpha=kw.get("ema_alpha", 0.02),
            warmup=kw.get("warmup", 50),
        )
    raise ValueError(f"unknown clutter remover: {name!r}")


# ============================================================
# Range-bin cropping
# ============================================================
def energy_per_bin(cir_iq: np.ndarray) -> np.ndarray:
    """Mean |I/Q|^2 per range bin, averaged over time, radar, antenna.

    cir_iq: (T, R, A, B, 2)  ->  (B,)
    """
    e = cir_iq[..., 0] ** 2 + cir_iq[..., 1] ** 2  # (T, R, A, B)
    return e.mean(axis=(0, 1, 2)).astype(np.float64)


def pick_bin_range(energy: np.ndarray, keep_fraction: float = 0.99) -> tuple[int, int]:
    """Pick (lo, hi) such that bins [lo, hi) contain ``keep_fraction``
    of the total energy and the remainder is split symmetrically between
    near-field and far-field tails.

    Returns Python ints; the half-open convention matches numpy slicing.
    """
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError("keep_fraction must be in (0, 1]")
    total = energy.sum()
    if total <= 0:
        return 0, len(energy)

    cdf = np.cumsum(energy) / total
    drop = (1.0 - keep_fraction) / 2.0
    lo = int(np.searchsorted(cdf, drop, side="left"))
    hi = int(np.searchsorted(cdf, 1.0 - drop, side="right"))
    lo = max(0, lo)
    hi = min(len(energy), max(hi, lo + 1))
    return lo, hi


# ============================================================
# Normalisation
# ============================================================
@dataclass
class NormStats:
    """Per-radar, per-antenna, per-channel mean/std for I/Q.

    Stats are computed on the *post-clutter-removal, post-crop* signal
    so that they reflect what the model actually sees.
    """
    mean: np.ndarray   # (R, A, 2), float32
    std:  np.ndarray   # (R, A, 2), float32

    def transform(self, x: np.ndarray) -> np.ndarray:
        """x: (T, R, A, B, 2) -> z-scored along axis 0 with broadcast."""
        # mean/std shape: (R, A, 2) -> broadcast to (1, R, A, 1, 2)
        m = self.mean[None, :, :, None, :]
        s = self.std[None, :, :, None, :]
        return ((x - m) / s).astype(np.float32, copy=False)


# ============================================================
# Top-level pipeline
# ============================================================
@dataclass
class Preprocessor:
    """End-to-end CIR preprocessor.

    Attributes
    ----------
    clutter_kind     : "running_mean" or "ema"
    clutter_kwargs   : dict of args passed to the remover
    bin_lo, bin_hi   : half-open range-bin window
    norm             : NormStats fit on the train split
    """
    clutter_kind: str
    clutter_kwargs: dict
    bin_lo: int
    bin_hi: int
    norm: NormStats

    # ---- per-frame transform -------------------------------------------------
    def transform(self, cir_iq: np.ndarray) -> np.ndarray:
        """Apply clutter removal -> crop -> normalise.

        cir_iq: (T, R, A, 120, 2) float32   ->   (T, R, A, B', 2) float32
        """
        remover = get_clutter_remover(self.clutter_kind, **self.clutter_kwargs)
        # Apply remover on I and Q jointly; remover treats axis 0 as time.
        x = remover(cir_iq.astype(np.float32, copy=False))
        x = x[:, :, :, self.bin_lo:self.bin_hi, :]
        x = self.norm.transform(x)
        return x

    # ---- save / load ---------------------------------------------------------
    def save(self, path: str | Path) -> None:
        np.savez(
            str(path),
            mean=self.norm.mean,
            std=self.norm.std,
            bin_lo=np.int32(self.bin_lo),
            bin_hi=np.int32(self.bin_hi),
            clutter_kind=np.array(self.clutter_kind),
            clutter_kwargs_json=np.array(_dict_to_json(self.clutter_kwargs)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Preprocessor":
        f = np.load(str(path), allow_pickle=False)
        return cls(
            clutter_kind=str(f["clutter_kind"]),
            clutter_kwargs=_json_to_dict(str(f["clutter_kwargs_json"])),
            bin_lo=int(f["bin_lo"]),
            bin_hi=int(f["bin_hi"]),
            norm=NormStats(mean=f["mean"], std=f["std"]),
        )

    # ---- fitting -------------------------------------------------------------
    @classmethod
    def fit(
        cls,
        train_window_paths: Sequence[str | Path],
        clutter: str = "ema",
        ema_alpha: float = 0.02,
        running_mean_window: int = 50,
        warmup: int = 50,
        energy_keep: float = 0.99,
        max_frames_for_stats: int | None = 30000,
        verbose: bool = True,
    ) -> "Preprocessor":
        """Fit the preprocessor on the training windows.

        Two passes over the training data:
          1. Clutter-remove each window, accumulate per-bin energy and
             a running tally of mean/var per (R, A, channel), restricted
             to the bins we'll eventually keep.
          2. After pass 1 we know (bin_lo, bin_hi); we then apply the
             crop and recompute mean/std on cropped data.

        For efficiency, mean/std are computed via Welford's algorithm
        over the *cropped* signal so we only do one accumulation.
        """
        if clutter == "ema":
            clutter_kwargs = {"ema_alpha": ema_alpha, "warmup": warmup}
        elif clutter == "running_mean":
            clutter_kwargs = {"window": running_mean_window}
        else:
            raise ValueError(f"unknown clutter kind: {clutter!r}")

        # ---- Pass 1: per-bin energy on cluttered signal ----
        if verbose:
            print(f"[fit] Pass 1/2: per-bin energy on {len(train_window_paths)} windows...")
        e_total = None
        for i, p in enumerate(train_window_paths):
            with np.load(p) as f:
                cir = f["radar_cir_iq"].astype(np.float32, copy=False)
            remover = get_clutter_remover(clutter, **clutter_kwargs)
            cir_clean = remover(cir)
            e = energy_per_bin(cir_clean)
            e_total = e if e_total is None else e_total + e
            if verbose:
                print(f"  [{i+1}/{len(train_window_paths)}] {Path(p).name} "
                      f"-> energy sum {e.sum():.3g}")
        bin_lo, bin_hi = pick_bin_range(e_total, keep_fraction=energy_keep)
        if verbose:
            print(f"[fit] Energy curve has {len(e_total)} bins; "
                  f"keeping [{bin_lo}, {bin_hi}) "
                  f"({bin_hi-bin_lo} bins, {energy_keep*100:.1f}% energy)")

        # ---- Pass 2: mean/std on (clutter-removed, cropped) signal ----
        if verbose:
            print(f"[fit] Pass 2/2: mean/std (Welford) on cropped data...")
        # Inspect a sample to learn shapes.
        with np.load(train_window_paths[0]) as f:
            R, A = f["radar_cir_iq"].shape[1:3]
        # Welford accumulators per (R, A, channel)
        n = 0
        mean = np.zeros((R, A, 2), dtype=np.float64)
        M2   = np.zeros((R, A, 2), dtype=np.float64)
        # Subsample by frame budget to keep this fast.
        frames_seen = 0
        for i, p in enumerate(train_window_paths):
            with np.load(p) as f:
                cir = f["radar_cir_iq"].astype(np.float32, copy=False)
            remover = get_clutter_remover(clutter, **clutter_kwargs)
            cir_clean = remover(cir)[:, :, :, bin_lo:bin_hi, :]
            T = cir_clean.shape[0]
            # If we already have enough samples, take just enough from this window.
            take = T
            if max_frames_for_stats is not None:
                remaining = max_frames_for_stats - frames_seen
                if remaining <= 0:
                    break
                take = min(T, remaining)
            chunk = cir_clean[:take]   # (take, R, A, B', 2)
            # We treat each (frame, range-bin) sample as i.i.d. for mean/std.
            # Welford-batch update along (frames, range-bins) axes.
            x = chunk.reshape(-1, R, A, 2).astype(np.float64, copy=False)
            n_chunk = x.shape[0] * (bin_hi - bin_lo)  # B' samples per frame
            # But we already flattened only the frame axis; flatten range-bin too:
            x_full = chunk.transpose(1, 2, 4, 0, 3).reshape(R, A, 2, -1)  # (R,A,2,N)
            n_chunk = x_full.shape[-1]
            chunk_mean = x_full.mean(axis=-1)
            chunk_M2 = ((x_full - chunk_mean[..., None]) ** 2).sum(axis=-1)
            # Welford merge
            delta = chunk_mean - mean
            new_n = n + n_chunk
            mean = mean + delta * (n_chunk / new_n)
            M2 = M2 + chunk_M2 + (delta ** 2) * (n * n_chunk / new_n)
            n = new_n
            frames_seen += take
            if verbose:
                print(f"  [{i+1}/{len(train_window_paths)}] frames_seen={frames_seen}")
            if max_frames_for_stats is not None and frames_seen >= max_frames_for_stats:
                break

        std = np.sqrt(M2 / max(n - 1, 1))
        # Floor std to avoid divide-by-zero in degenerate channels.
        std = np.maximum(std, 1e-6)
        norm = NormStats(mean=mean.astype(np.float32), std=std.astype(np.float32))

        if verbose:
            print(f"[fit] mean range [{mean.min():.3g}, {mean.max():.3g}], "
                  f"std range [{std.min():.3g}, {std.max():.3g}]")

        return cls(
            clutter_kind=clutter,
            clutter_kwargs=clutter_kwargs,
            bin_lo=bin_lo, bin_hi=bin_hi,
            norm=norm,
        )

def _dict_to_json(d: dict) -> str:
    import json
    return json.dumps(d, sort_keys=True)


def _json_to_dict(s: str) -> dict:
    import json
    return json.loads(s)


__all__ = [
    "RunningMeanClutterRemover",
    "EMAClutterRemover",
    "get_clutter_remover",
    "energy_per_bin",
    "pick_bin_range",
    "NormStats",
    "Preprocessor",
]
