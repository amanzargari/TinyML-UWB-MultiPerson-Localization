"""Unit tests for model.py. Run with: python test_model.py"""
import os, sys, tempfile
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, '.')

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from model import (
    build_model, convert_to_tflite, get_default_losses_and_weights,
    MODEL_VARIANTS,
    T_CONTEXT, N_RADARS, N_ANT, N_BINS, N_IQ,
    GRID_H, GRID_W, COUNT_CLASSES,
)

ESP32_FLASH_KB = 800
ESP32_SRAM_KB = 400


def test_variants_registered():
    assert set(MODEL_VARIANTS.keys()) == {"small", "medium", "large"}
    print("PASS variants_registered")


def test_build_each_variant():
    for name in MODEL_VARIANTS:
        m = build_model(name)
        assert m.input_shape == (None, T_CONTEXT, N_RADARS, N_ANT, N_BINS, N_IQ)
        # output_names ordered, output_shape is list-of-tuples
        out_by_name = dict(zip(m.output_names, [tuple(o.shape) for o in m.outputs]))
        assert out_by_name["heatmap"] == (None, GRID_H, GRID_W, 1), f"got {out_by_name}"
        assert out_by_name["count"]   == (None, COUNT_CLASSES),     f"got {out_by_name}"
    print("PASS build_each_variant")


def test_forward_pass():
    """Output shapes correct, sigmoid in [0,1], softmax sums to 1."""
    m = build_model("small")
    x = np.random.randn(2, T_CONTEXT, N_RADARS, N_ANT, N_BINS, N_IQ).astype(np.float32)
    out = m(x)
    # Output is a dict for our Functional model
    if isinstance(out, dict):
        hm, cnt = out["heatmap"].numpy(), out["count"].numpy()
    else:
        # Fallback: list ordered by m.output_names
        d = dict(zip(m.output_names, out))
        hm, cnt = d["heatmap"].numpy(), d["count"].numpy()
    assert hm.shape == (2, GRID_H, GRID_W, 1)
    assert cnt.shape == (2, COUNT_CLASSES)
    assert hm.min() >= 0.0 and hm.max() <= 1.0
    assert np.allclose(cnt.sum(axis=1), 1.0, atol=1e-5)
    print("PASS forward_pass")


def test_invalid_variant_raises():
    try:
        build_model("xlarge")
    except ValueError as e:
        assert "xlarge" in str(e)
        print("PASS invalid_variant_raises")
        return
    raise AssertionError("expected ValueError")


def test_compile_with_default_losses():
    m = build_model("small")
    losses, weights, metrics = get_default_losses_and_weights(lambda_count=0.1)
    m.compile(optimizer="adam", loss=losses, loss_weights=weights, metrics=metrics)
    # Run one train step to confirm compatibility
    x = np.random.randn(4, T_CONTEXT, N_RADARS, N_ANT, N_BINS, N_IQ).astype(np.float32)
    y_hm = np.random.rand(4, GRID_H, GRID_W, 1).astype(np.float32)
    y_cnt = np.random.randint(0, COUNT_CLASSES, size=(4,)).astype(np.int32)
    history = m.fit(x, {"heatmap": y_hm, "count": y_cnt}, epochs=1, verbose=0)
    assert "loss" in history.history
    print("PASS compile_with_default_losses")


def test_tflite_conversion_int8():
    """Each variant converts to INT8 TFLite within size budget."""
    def rep():
        for _ in range(20):
            yield [np.random.randn(1, T_CONTEXT, N_RADARS, N_ANT, N_BINS, N_IQ).astype(np.float32)]

    sizes = {}
    for name in MODEL_VARIANTS:
        m = build_model(name)
        # Warm BN before conversion
        np.random.seed(42)
        for _ in range(3):
            _ = m(np.random.randn(4, T_CONTEXT, N_RADARS, N_ANT, N_BINS, N_IQ).astype(np.float32),
                  training=True)
        with tempfile.TemporaryDirectory() as td:
            path = f"{td}/{name}.tflite"
            size_bytes = convert_to_tflite(m, name, rep, path, quantize_int8=True)
            kb = size_bytes / 1024
            sizes[name] = kb
            assert kb < ESP32_FLASH_KB, f"{name} INT8 size {kb:.1f}KB >= {ESP32_FLASH_KB}KB"
    print(f"PASS tflite_conversion_int8 (sizes KB: {sizes})")


def test_tflite_runs_inference():
    """Round-trip: convert and run inference on the INT8 model."""
    m = build_model("small")
    np.random.seed(42)
    for _ in range(3):
        _ = m(np.random.randn(4, T_CONTEXT, N_RADARS, N_ANT, N_BINS, N_IQ).astype(np.float32),
              training=True)

    def rep():
        for _ in range(20):
            yield [np.random.randn(1, T_CONTEXT, N_RADARS, N_ANT, N_BINS, N_IQ).astype(np.float32)]

    with tempfile.TemporaryDirectory() as td:
        path = f"{td}/m.tflite"
        convert_to_tflite(m, "small", rep, path, quantize_int8=True)

        interp = tf.lite.Interpreter(model_path=path)
        interp.allocate_tensors()
        in_det = interp.get_input_details()[0]
        out_dets = interp.get_output_details()
        assert in_det["shape"].tolist() == [1, T_CONTEXT, N_RADARS, N_ANT, N_BINS, N_IQ]
        assert in_det["dtype"] == np.int8

        # Build an INT8 input from a representative float input
        x = np.random.randn(1, T_CONTEXT, N_RADARS, N_ANT, N_BINS, N_IQ).astype(np.float32)
        scale, zp = in_det["quantization"]
        x_q = np.clip(np.round(x / scale + zp), -128, 127).astype(np.int8)
        interp.set_tensor(in_det["index"], x_q)
        interp.invoke()
        # Each output should produce a result of correct shape
        for od in out_dets:
            tensor = interp.get_tensor(od["index"])
            assert tensor.shape[0] == 1
    print("PASS tflite_runs_inference")


def test_no_forbidden_ops_int8():
    """The INT8 model contains no forbidden ops (LSTM/GRU/RNN)."""
    forbidden = {"LSTM", "GRU", "RNN", "UNIDIRECTIONAL_SEQUENCE_LSTM",
                 "BIDIRECTIONAL_SEQUENCE_LSTM"}
    m = build_model("small")
    np.random.seed(42)
    for _ in range(3):
        _ = m(np.random.randn(4, T_CONTEXT, N_RADARS, N_ANT, N_BINS, N_IQ).astype(np.float32),
              training=True)

    def rep():
        for _ in range(20):
            yield [np.random.randn(1, T_CONTEXT, N_RADARS, N_ANT, N_BINS, N_IQ).astype(np.float32)]

    with tempfile.TemporaryDirectory() as td:
        path = f"{td}/m.tflite"
        convert_to_tflite(m, "small", rep, path, quantize_int8=True)
        interp = tf.lite.Interpreter(model_path=path)
        interp.allocate_tensors()
        ops = set()
        for op in interp._get_ops_details():
            ops.add(op["op_name"])
        # No forbidden ops should appear
        bad = ops & forbidden
        assert not bad, f"Found forbidden ops: {bad}"
        print(f"PASS no_forbidden_ops_int8 (ops used: {sorted(ops)})")


if __name__ == "__main__":
    tests = [
        test_variants_registered,
        test_build_each_variant,
        test_forward_pass,
        test_invalid_variant_raises,
        test_compile_with_default_losses,
        test_tflite_conversion_int8,
        test_tflite_runs_inference,
        test_no_forbidden_ops_int8,
    ]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} tests passed.")
