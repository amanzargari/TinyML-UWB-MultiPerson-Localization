"""
convert_to_litert.py
====================

PyTorch -> TFLite conversion helper using ``litert_torch`` (formerly
``ai_edge_torch``).

Two conversion paths:
  - ``convert_float32(model, sample_inputs, output_path)``
  - ``convert_int8(model, sample_inputs, representative_dataset_fn, output_path)``

The INT8 path uses PT2E (PyTorch 2 Export-based) quantization, which:
  1. Exports the model graph via ``torch.export``.
  2. Annotates the graph with quantization specs via ``PT2EQuantizer``.
  3. Calibrates the activation ranges using a representative dataset.
  4. Converts to a quantized graph via ``convert_pt2e``.
  5. Lowers to TFLite via ``litert_torch.convert``.

The output is a deployable .tflite file with INT8 weights and activations,
suitable for ESP32-S3 via TensorFlow Lite for Microcontrollers.

Author: Sepehr (TinyML UWB project, 2026)
"""
from __future__ import annotations

import os
from typing import Callable, Iterable

import torch


# ============================================================
# Float32 conversion
# ============================================================
def convert_float32(
    model: torch.nn.Module,
    sample_inputs: tuple[torch.Tensor, ...],
    output_path: str,
) -> int:
    """Convert a PyTorch model to a float32 TFLite file.

    Parameters
    ----------
    model : torch.nn.Module
        The model to convert. Should already be in eval mode.
    sample_inputs : tuple of torch.Tensor
        A representative input tuple. Used for tracing and shape inference.
        Must include the batch dimension (typically batch_size=1).
    output_path : str
        Path to write the .tflite file.

    Returns
    -------
    int : number of bytes written.
    """
    import litert_torch

    model.eval()
    edge_model = litert_torch.convert(model, sample_inputs)
    edge_model.export(output_path)
    return os.path.getsize(output_path)


# ============================================================
# INT8 conversion via PT2E quantization
# ============================================================
def convert_int8(
    model: torch.nn.Module,
    sample_inputs: tuple[torch.Tensor, ...],
    representative_dataset_fn: Callable[[], Iterable[tuple[torch.Tensor, ...]]],
    output_path: str,
    n_calibration: int = 20,
) -> int:
    """Convert a PyTorch model to an INT8 TFLite file via PT2E quantization.

    Parameters
    ----------
    model : torch.nn.Module
        Model to quantize. Should be in eval mode and have all BN stats
        well-calibrated (i.e. trained, or warmed up by some forward passes
        with training=True).
    sample_inputs : tuple of torch.Tensor
        Representative input tuple for tracing. Batch size 1 recommended.
    representative_dataset_fn : callable
        A no-arg callable returning an iterable of input tuples. Each
        yielded item must be a tuple matching the structure of
        ``sample_inputs`` (e.g. ``(tensor,)``). Used for activation-range
        calibration.
    output_path : str
        Path to write the .tflite file.
    n_calibration : int
        Maximum number of calibration samples to consume from the
        representative dataset.

    Returns
    -------
    int : number of bytes written.
    """
    import litert_torch
    from litert_torch.quantize.pt2e_quantizer import (
        PT2EQuantizer, get_symmetric_quantization_config,
    )
    from litert_torch.quantize.quant_config import QuantConfig
    from torchao.quantization.pt2e.quantize_pt2e import (
        prepare_pt2e, convert_pt2e,
    )

    model.eval()

    # 1. Build the PT2E quantizer with a symmetric per-channel config.
    quantizer = PT2EQuantizer().set_global(
        get_symmetric_quantization_config(
            is_per_channel=True,
            is_dynamic=False,
        )
    )

    # 2. Export the model graph via torch.export.
    m_exported = torch.export.export(model, sample_inputs).module()

    # 3. Prepare for PT2E quantization (inserts observer modules).
    m_prepared = prepare_pt2e(m_exported, quantizer)

    # 4. Calibrate by running representative inputs through the prepared model.
    with torch.no_grad():
        for i, batch in enumerate(representative_dataset_fn()):
            if i >= n_calibration:
                break
            if not isinstance(batch, (tuple, list)):
                batch = (batch,)
            m_prepared(*batch)
            del batch

    # 5. Convert to a quantized graph.
    m_quantized = convert_pt2e(m_prepared, fold_quantize=False)
    del m_prepared

    # 6. Lower to TFLite with the quantization config attached.
    edge_int8 = litert_torch.convert(
        m_quantized, sample_inputs,
        quant_config=QuantConfig(pt2e_quantizer=quantizer),
    )
    del m_quantized

    edge_int8.export(output_path)
    return os.path.getsize(output_path)


# ============================================================
# Inspection helpers
# ============================================================
def inspect_tflite(path: str) -> dict:
    """Inspect a .tflite file: ops used, largest tensor, arena estimate.

    Requires tensorflow installed (only for the inspection tooling --
    not used during conversion).

    Returns
    -------
    dict with keys:
        size_kb         : float
        op_counts       : dict[str, int]
        largest_tensor  : (shape: tuple, bytes: int)
        arena_estimate_kb : float (= 2 * largest tensor)
    """
    import numpy as np
    import tensorflow as tf
    from collections import Counter

    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()

    # Collect ops
    ops = []
    if hasattr(interp, "_get_ops_details"):
        for op in interp._get_ops_details():
            ops.append(op["op_name"])

    # Find largest tensor
    tensors = interp.get_tensor_details()
    largest_bytes = 0
    largest_shape = None
    for t in tensors:
        sh = t["shape"]
        if sh.size == 0:
            continue
        nbytes = int(sh.prod()) * np.dtype(t["dtype"]).itemsize
        if nbytes > largest_bytes:
            largest_bytes = nbytes
            largest_shape = tuple(sh.tolist())

    return {
        "size_kb": os.path.getsize(path) / 1024,
        "op_counts": dict(Counter(ops)),
        "largest_tensor": (largest_shape, largest_bytes),
        "arena_estimate_kb": largest_bytes * 2 / 1024,
    }


__all__ = [
    "convert_float32",
    "convert_int8",
    "inspect_tflite",
]
