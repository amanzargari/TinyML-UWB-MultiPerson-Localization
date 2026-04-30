"""
model.py
========

Hybrid heatmap + count-head model for multi-person UWB localization.

Architecture (all ops TFLM-supported):
    Input:  (T_CONTEXT, N_RADARS, N_ANT, N_BINS, N_IQ)
    Stem:   Reshape to (T_CONTEXT, N_BINS, N_RADARS*N_ANT*N_IQ)
            i.e. a "2D image" with H=time, W=range, C=radar*ant*iq.
            The first conv mixes radars/antennas/iq channels.
    Encoder:
            Stage 0: DepthwiseConv2D (T,1) + BN + ReLU  -- collapses time
            Stage 1: Conv2D 1x5 + BN + ReLU
            Stage 2: Conv2D 1x5 stride (1,2) + BN + ReLU
            Stage 3: Conv2D 1x3 stride (1,2) + BN + ReLU
            GlobalAveragePooling2D -> flat feature
    Heatmap decoder:
            Dense -> Reshape (BH, BW, c_b)  [STATIC shape]
            Conv2D 1x1 expand -> c0
            UpSampling2D 2x + Conv2D 3x3 + BN + ReLU -> (18, 12, c1)
            UpSampling2D 2x + Conv2D 3x3 + BN + ReLU -> (36, 24, c2)
            Conv2D 1x1 + sigmoid -> (36, 24, 1)
    Count head:
            Dense + ReLU + Dense + softmax -> (5,)

Three size variants are provided (small/medium/large) selectable via
``build_model(variant=...)``.

Author: Sepehr (TinyML UWB project, 2026)
"""
from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf
from tensorflow.keras import layers, Model

# ============================================================
# Constants
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
BOTTLENECK_H = GRID_H // 4
BOTTLENECK_W = GRID_W // 4


# ============================================================
# Variant definitions
# ============================================================
@dataclass(frozen=True)
class Variant:
    name: str
    enc_filters: tuple[int, int, int, int]   # 4 conv stages
    feature_width: int                        # Dense projection after GAP
    bottleneck_chan: int                      # c_b: small ch at static reshape
    decoder_channels: tuple[int, int, int]   # c0, c1, c2
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
# Model builder
# ============================================================
def build_model(variant: str = "medium",
                batch_size: int | None = None) -> Model:
    """Build the hybrid model.

    Parameters
    ----------
    variant : "small", "medium", or "large".
    batch_size : if not None, bake the batch dimension into the input shape.
        Use ``batch_size=None`` for training (Keras's default dynamic batch)
        and ``batch_size=1`` when building a model for TFLite conversion --
        the converter requires a static input shape to fold all variables
        and avoid `tf.Conv2D` fallbacks.
    """
    if variant not in MODEL_VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}. "
                         f"Choose from {list(MODEL_VARIANTS)}.")
    v = MODEL_VARIANTS[variant]

    # ============ Input ============
    if batch_size is None:
        inputs = layers.Input(
            shape=(T_CONTEXT, N_RADARS, N_ANT, N_BINS, N_IQ),
            name="cir_stack",
        )
    else:
        inputs = layers.Input(
            batch_shape=(batch_size, T_CONTEXT, N_RADARS, N_ANT, N_BINS, N_IQ),
            name="cir_stack",
        )

    # ============ Stem: rearrange to 2D image (T, B, R*A*IQ) ============
    # axes after batch: T(0) R(1) A(2) B(3) IQ(4)
    # want order:        T,    B,   R,   A,   IQ  -> permute (0, 3, 1, 2, 4)
    x = layers.Permute((1, 4, 2, 3, 5), name="permute_to_TBRAIQ")(inputs)
    x = layers.Reshape(
        (T_CONTEXT, N_BINS, INPUT_CHANNELS),
        name="reshape_to_image",
    )(x)
    # Now: (T=8, B=105, C=36)

    f0, f1, f2, f3 = v.enc_filters

    # ============ Stage 0: depthwise temporal collapse ============
    x = layers.DepthwiseConv2D(
        kernel_size=(T_CONTEXT, 1),
        depth_multiplier=1,
        padding="valid",
        use_bias=False,
        name="dw_time",
    )(x)
    # Shape: (1, 105, 36)
    x = layers.BatchNormalization(name="bn_dw")(x)
    x = layers.ReLU(name="relu_dw")(x)

    x = layers.Conv2D(f0, (1, 1), padding="same", use_bias=False, name="conv0_pw")(x)
    x = layers.BatchNormalization(name="bn0")(x)
    x = layers.ReLU(name="relu0")(x)

    # ============ Stage 1: 1x5 conv ============
    x = layers.Conv2D(f1, (1, 5), padding="same", use_bias=False, name="conv1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)

    # ============ Stage 2: 1x5 stride 2 ============
    x = layers.Conv2D(f2, (1, 5), strides=(1, 2), padding="same",
                      use_bias=False, name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.ReLU(name="relu2")(x)

    # ============ Stage 3: 1x3 stride 2 ============
    x = layers.Conv2D(f3, (1, 3), strides=(1, 2), padding="same",
                      use_bias=False, name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.ReLU(name="relu3")(x)

    # ============ GAP + Dense ============
    features = layers.GlobalAveragePooling2D(name="gap")(x)
    features = layers.Dense(v.feature_width, use_bias=True, name="feat_dense")(features)
    features = layers.ReLU(name="feat_relu")(features)

    # ============ Heatmap decoder ============
    c_b = v.bottleneck_chan
    c0, c1, c2 = v.decoder_channels

    h = layers.Dense(BOTTLENECK_H * BOTTLENECK_W * c_b, use_bias=True,
                     name="dec_proj")(features)
    h = layers.ReLU(name="dec_proj_relu")(h)
    h = layers.Reshape((BOTTLENECK_H, BOTTLENECK_W, c_b), name="dec_reshape")(h)

    h = layers.Conv2D(c0, 1, padding="same", use_bias=False, name="dec_expand")(h)
    h = layers.BatchNormalization(name="dec_bn0")(h)
    h = layers.ReLU(name="dec_relu0")(h)

    h = layers.UpSampling2D(size=(2, 2), interpolation="nearest", name="dec_up1")(h)
    h = layers.Conv2D(c1, 3, padding="same", use_bias=False, name="dec_conv1")(h)
    h = layers.BatchNormalization(name="dec_bn1")(h)
    h = layers.ReLU(name="dec_relu1")(h)

    h = layers.UpSampling2D(size=(2, 2), interpolation="nearest", name="dec_up2")(h)
    h = layers.Conv2D(c2, 3, padding="same", use_bias=False, name="dec_conv2")(h)
    h = layers.BatchNormalization(name="dec_bn2")(h)
    h = layers.ReLU(name="dec_relu2")(h)

    heatmap_out = layers.Conv2D(1, 1, padding="same", activation="sigmoid",
                                name="heatmap")(h)

    # ============ Count head ============
    c = layers.Dense(v.count_hidden, use_bias=True, name="count_dense")(features)
    c = layers.ReLU(name="count_relu")(c)
    count_out = layers.Dense(COUNT_CLASSES, activation="softmax", name="count")(c)

    return Model(
        inputs=inputs,
        outputs={"heatmap": heatmap_out, "count": count_out},
        name=f"uwb_localization_{variant}",
    )


# ============================================================
# TFLite conversion helper
# ============================================================
def convert_to_tflite(
    trained_model: Model,
    variant: str,
    representative_dataset_fn,
    output_path: str,
    quantize_int8: bool = True,
) -> int:
    """Convert a trained model to TFLite, optionally INT8-quantized.

    Workflow (the only reliable recipe we found):
        1. Build a fresh inference-only model with batch_size=1.
        2. Copy weights from the trained (dynamic-batch) model.
        3. Save .keras then reload (bakes BN moving stats as constants).
        4. Convert via tf.lite.TFLiteConverter.from_keras_model.

    Parameters
    ----------
    trained_model : the model produced by ``build_model(variant)`` after
        training. Must match ``variant``.
    variant : the variant name used to build trained_model.
    representative_dataset_fn : callable yielding lists of one float32 input
        of shape (1, T_CONTEXT, N_RADARS, N_ANT, N_BINS, N_IQ). Used for
        INT8 calibration.
    output_path : .tflite file to write.
    quantize_int8 : if True, full INT8 quantization. Otherwise float32.

    Returns
    -------
    Number of bytes written.
    """
    import tempfile

    # 1. Fresh inference-only model with fixed batch
    inf_model = build_model(variant, batch_size=1)
    inf_model.set_weights(trained_model.get_weights())

    # 2. Run a dummy forward pass to ensure all internal state is realized
    import numpy as np
    _ = inf_model(np.zeros((1, T_CONTEXT, N_RADARS, N_ANT, N_BINS, N_IQ),
                           dtype=np.float32),
                  training=False)

    # 3. Save & reload to bake BN stats
    with tempfile.TemporaryDirectory() as td:
        keras_path = f"{td}/inf_model.keras"
        inf_model.save(keras_path)
        reloaded = tf.keras.models.load_model(keras_path)

        # 4. Convert
        converter = tf.lite.TFLiteConverter.from_keras_model(reloaded)
        if quantize_int8:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset_fn
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        tflite_bytes = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_bytes)

    return len(tflite_bytes)


# ============================================================
# Compile helper
# ============================================================
def get_default_losses_and_weights(lambda_count: float = 0.1):
    """Default loss / weight / metrics specs for training."""
    losses = {
        "heatmap": tf.keras.losses.BinaryCrossentropy(),
        "count":   tf.keras.losses.SparseCategoricalCrossentropy(),
    }
    loss_weights = {"heatmap": 1.0, "count": float(lambda_count)}
    metrics = {
        "heatmap": [tf.keras.metrics.BinaryAccuracy(name="hm_acc")],
        "count":   [tf.keras.metrics.SparseCategoricalAccuracy(name="cnt_acc")],
    }
    return losses, loss_weights, metrics


__all__ = [
    "MODEL_VARIANTS", "Variant", "build_model",
    "convert_to_tflite",
    "get_default_losses_and_weights",
    "T_CONTEXT", "N_RADARS", "N_ANT", "N_BINS", "N_IQ",
    "GRID_H", "GRID_W", "COUNT_CLASSES",
]
