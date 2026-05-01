"""
model_torch.py
==============

PyTorch implementation of the hybrid heatmap + count-head model for
multi-person UWB localization, designed for conversion to TFLite via
``litert_torch`` (formerly ``ai_edge_torch``).

This is a 1-for-1 architectural port of the TensorFlow ``model.py``,
preserving the same channel widths, op sequence, and inference semantics.
The two notable differences relative to the TF version are:

  1. NCHW tensor layout (PyTorch convention).
  2. GAP is implemented as ``F.adaptive_avg_pool2d(x, 1).flatten(1)``
     instead of ``x.mean(dim=(2, 3))``, so that it compiles to a single
     ``MEAN`` op in TFLite rather than ``SUM`` + ``MUL`` (the latter is
     not on every TFLM build's default OpResolver).

Three size variants (small / medium / large) are provided. After INT8
quantization with ``litert_torch``, all three fit the ESP32-S3 budgets
of 800 KB flash and 400 KB tensor arena (verified empirically).

The TFLite conversion recipe is in ``convert_to_litert.py``.

Author: Sepehr (TinyML UWB project, 2026)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Constants -- match heatmap.py and preprocessing.py
# ============================================================
T_CONTEXT = 8
N_RADARS = 6
N_ANT = 3
N_BINS = 105
N_IQ = 2
INPUT_CHANNELS = N_RADARS * N_ANT * N_IQ  # 36

GRID_H = 36
GRID_W = 24
COUNT_CLASSES = 5

# Bottleneck spatial: 9 x 6, doubled twice -> 36 x 24
BOTTLENECK_H = GRID_H // 4   # 9
BOTTLENECK_W = GRID_W // 4   # 6


# ============================================================
# Variant definitions
# ============================================================
@dataclass(frozen=True)
class Variant:
    name: str
    enc_filters: tuple[int, int, int, int]
    feature_width: int
    bottleneck_chan: int
    decoder_channels: tuple[int, int, int]
    count_hidden: int


# Targets:
#   small  ~150 KB INT8
#   medium ~400 KB INT8
#   large  ~700 KB INT8
MODEL_VARIANTS: dict[str, Variant] = {
    "small": Variant(
        name="small",
        enc_filters=(32, 48, 64, 96),
        feature_width=96,
        bottleneck_chan=8,
        decoder_channels=(48, 32, 16),
        count_hidden=48,
    ),
    "medium": Variant(
        name="medium",
        enc_filters=(48, 80, 112, 160),
        feature_width=160,
        bottleneck_chan=12,
        decoder_channels=(80, 56, 28),
        count_hidden=80,
    ),
    "large": Variant(
        name="large",
        enc_filters=(64, 112, 160, 224),
        feature_width=224,
        bottleneck_chan=16,
        decoder_channels=(112, 80, 40),
        count_hidden=128,
    ),
}


# ============================================================
# Model
# ============================================================
class UWBLocalizer(nn.Module):
    """Hybrid heatmap + count head for multi-person UWB localization.

    Input  : (B, T_CONTEXT, N_RADARS, N_ANT, N_BINS, N_IQ) float32
    Outputs: tuple ``(heatmap, count)``
        heatmap : (B, 1, GRID_H, GRID_W) sigmoid presence map
        count   : (B, COUNT_CLASSES)     softmax over {0, 1, 2, 3, 4}
    """

    def __init__(self, variant: str = "medium"):
        super().__init__()
        if variant not in MODEL_VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}. "
                             f"Choose from {list(MODEL_VARIANTS)}.")
        self.variant_name = variant
        v = MODEL_VARIANTS[variant]
        self.variant = v

        f0, f1, f2, f3 = v.enc_filters
        c_b = v.bottleneck_chan
        c0, c1, c2 = v.decoder_channels

        # ---------- Encoder ----------
        # Stage 0: depthwise temporal collapse over H=T axis
        # Input after permute+reshape:  (B, C=36, H=8, W=105)
        # Output of dw_time:            (B, C=36, H=1, W=105)
        self.dw_time = nn.Conv2d(
            in_channels=INPUT_CHANNELS,
            out_channels=INPUT_CHANNELS,
            kernel_size=(T_CONTEXT, 1),
            groups=INPUT_CHANNELS,        # depthwise
            bias=False,
        )
        self.bn_dw = nn.BatchNorm2d(INPUT_CHANNELS)

        # Pointwise mix to f0 channels
        self.conv0_pw = nn.Conv2d(INPUT_CHANNELS, f0, kernel_size=1, bias=False)
        self.bn0 = nn.BatchNorm2d(f0)

        # Stage 1: 1x5 conv along the range axis
        self.conv1 = nn.Conv2d(f0, f1, kernel_size=(1, 5), padding=(0, 2), bias=False)
        self.bn1 = nn.BatchNorm2d(f1)

        # Stage 2: 1x5 stride-2 (range halves)
        self.conv2 = nn.Conv2d(f1, f2, kernel_size=(1, 5), stride=(1, 2),
                               padding=(0, 2), bias=False)
        self.bn2 = nn.BatchNorm2d(f2)

        # Stage 3: 1x3 stride-2
        self.conv3 = nn.Conv2d(f2, f3, kernel_size=(1, 3), stride=(1, 2),
                               padding=(0, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(f3)

        # GAP + Dense
        self.feat_dense = nn.Linear(f3, v.feature_width)

        # ---------- Heatmap decoder ----------
        # Project flat features -> (BH, BW, c_b) via Dense + Reshape
        self.dec_proj = nn.Linear(v.feature_width, BOTTLENECK_H * BOTTLENECK_W * c_b)

        # 1x1 expand c_b -> c0
        self.dec_expand = nn.Conv2d(c_b, c0, kernel_size=1, bias=False)
        self.dec_bn0 = nn.BatchNorm2d(c0)

        # Up 2x + 3x3 conv: (9, 6, c0) -> (18, 12, c1)
        self.dec_conv1 = nn.Conv2d(c0, c1, kernel_size=3, padding=1, bias=False)
        self.dec_bn1 = nn.BatchNorm2d(c1)

        # Up 2x + 3x3 conv: (18, 12, c1) -> (36, 24, c2)
        self.dec_conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False)
        self.dec_bn2 = nn.BatchNorm2d(c2)

        # Output 1x1
        self.heatmap_out = nn.Conv2d(c2, 1, kernel_size=1)

        # ---------- Count head ----------
        self.count_dense = nn.Linear(v.feature_width, v.count_hidden)
        self.count_out = nn.Linear(v.count_hidden, COUNT_CLASSES)

        # Cache c_b for forward
        self._c_b = c_b

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        x: (B, T, R, A, B, IQ) float32
        returns: (heatmap, count) where
            heatmap shape = (B, 1, GRID_H, GRID_W)
            count   shape = (B, COUNT_CLASSES)
        """
        B, T, R, A, BB, IQ = x.shape

        # Stem: rearrange to NCHW with C=R*A*IQ, H=T, W=B
        # Permute: (B, R, A, IQ, T, BB)
        x = x.permute(0, 2, 3, 5, 1, 4).contiguous()
        x = x.reshape(B, INPUT_CHANNELS, T, BB)         # (B, 36, 8, 105)

        # ---- Stage 0: depthwise temporal collapse ----
        x = self.dw_time(x)                             # (B, 36, 1, 105)
        x = self.bn_dw(x)
        x = F.relu(x)
        x = self.conv0_pw(x)                            # (B, f0, 1, 105)
        x = self.bn0(x)
        x = F.relu(x)

        # ---- Stage 1-3: range conv stack ----
        x = F.relu(self.bn1(self.conv1(x)))             # (B, f1, 1, 105)
        x = F.relu(self.bn2(self.conv2(x)))             # (B, f2, 1, 53)
        x = F.relu(self.bn3(self.conv3(x)))             # (B, f3, 1, 27)

        # ---- GAP + Dense ----
        # adaptive_avg_pool2d compiles to MEAN in TFLite (cleaner than .mean()).
        x = F.adaptive_avg_pool2d(x, 1)                 # (B, f3, 1, 1)
        x = x.flatten(1)                                # (B, f3)
        feat = F.relu(self.feat_dense(x))               # (B, F)

        # ---- Heatmap decoder ----
        h = F.relu(self.dec_proj(feat))                 # (B, BH*BW*c_b)
        h = h.reshape(B, self._c_b, BOTTLENECK_H, BOTTLENECK_W)

        h = F.relu(self.dec_bn0(self.dec_expand(h)))    # (B, c0, 9, 6)
        h = F.interpolate(h, scale_factor=2.0, mode="nearest")  # (B, c0, 18, 12)
        h = F.relu(self.dec_bn1(self.dec_conv1(h)))     # (B, c1, 18, 12)
        h = F.interpolate(h, scale_factor=2.0, mode="nearest")  # (B, c1, 36, 24)
        h = F.relu(self.dec_bn2(self.dec_conv2(h)))     # (B, c2, 36, 24)
        heatmap = torch.sigmoid(self.heatmap_out(h))    # (B, 1, 36, 24)

        # ---- Count head ----
        c = F.relu(self.count_dense(feat))
        count = F.softmax(self.count_out(c), dim=1)

        return heatmap, count


# ============================================================
# Convenience builder mirroring the TF API
# ============================================================
def build_model(variant: str = "medium") -> UWBLocalizer:
    """Build a fresh model. Mirrors the TF build_model signature."""
    return UWBLocalizer(variant=variant)


# ============================================================
# Default loss specs (for use in training notebook)
# ============================================================
def get_default_losses(lambda_count: float = 0.1):
    """Return (heatmap_loss_fn, count_loss_fn, lambda_count).

    heatmap loss: per-pixel BCE on a [0, 1] target.
    count loss:   cross-entropy on integer class labels.

    Total loss is built externally as:
        L = heatmap_bce + lambda_count * count_ce
    """
    bce = nn.BCELoss()
    ce  = nn.CrossEntropyLoss()
    return bce, ce, float(lambda_count)


__all__ = [
    "MODEL_VARIANTS", "Variant",
    "UWBLocalizer", "build_model",
    "get_default_losses",
    "T_CONTEXT", "N_RADARS", "N_ANT", "N_BINS", "N_IQ",
    "INPUT_CHANNELS",
    "GRID_H", "GRID_W", "COUNT_CLASSES",
    "BOTTLENECK_H", "BOTTLENECK_W",
]
