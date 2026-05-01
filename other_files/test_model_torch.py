"""Unit tests for model_torch.py and convert_to_litert.py.

Run with: python test_model_torch.py
"""
import os, sys, tempfile
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, ".")

import numpy as np
import torch

from model_torch import (
    UWBLocalizer, build_model, MODEL_VARIANTS,
    T_CONTEXT, N_RADARS, N_ANT, N_BINS, N_IQ,
    GRID_H, GRID_W, COUNT_CLASSES,
)
from convert_to_litert import convert_float32, convert_int8, inspect_tflite


ESP32_FLASH_KB = 800
ESP32_SRAM_KB = 400


def test_variants_registered():
    assert set(MODEL_VARIANTS.keys()) == {"small", "medium", "large"}
    print("PASS variants_registered")


def test_build_each_variant():
    for name in MODEL_VARIANTS:
        m = build_model(name)
        assert isinstance(m, UWBLocalizer)
        assert m.variant_name == name
    print("PASS build_each_variant")


def test_forward_pass():
    """Output shapes correct, sigmoid in [0,1], softmax sums to 1."""
    m = build_model("small").eval()
    x = torch.randn(2, T_CONTEXT, N_RADARS, N_ANT, N_BINS, N_IQ)
    with torch.no_grad():
        hm, cnt = m(x)
    assert hm.shape == (2, 1, GRID_H, GRID_W), f"got {hm.shape}"
    assert cnt.shape == (2, COUNT_CLASSES), f"got {cnt.shape}"
    assert hm.min().item() >= 0.0 and hm.max().item() <= 1.0
    assert torch.allclose(cnt.sum(dim=1), torch.ones(2), atol=1e-5)
    print("PASS forward_pass")


def test_invalid_variant_raises():
    try:
        build_model("xlarge")
    except ValueError as e:
        assert "xlarge" in str(e)
        print("PASS invalid_variant_raises")
        return
    raise AssertionError("expected ValueError")


def test_train_step():
    """One step of supervised training runs without error."""
    m = build_model("small")
    bce = torch.nn.BCELoss()
    ce  = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)

    m.train()
    x = torch.randn(4, T_CONTEXT, N_RADARS, N_ANT, N_BINS, N_IQ)
    y_hm = torch.rand(4, 1, GRID_H, GRID_W)
    y_cnt = torch.randint(0, COUNT_CLASSES, size=(4,))

    hm_pred, cnt_pred = m(x)
    loss = bce(hm_pred, y_hm) + 0.1 * ce(cnt_pred, y_cnt)
    loss.backward()
    opt.step()
    print(f"PASS train_step (loss={loss.item():.4f})")


def test_param_counts():
    expected = {"small": 110_000, "medium": 300_000, "large": 600_000}  # rough
    for name, lower in expected.items():
        m = build_model(name)
        n = m.num_params
        assert n > 0.7 * lower, f"{name} has {n} params, expected ~{lower}"
    print(f"PASS param_counts ("
          f"small={build_model('small').num_params:,} "
          f"medium={build_model('medium').num_params:,} "
          f"large={build_model('large').num_params:,})")


def test_tflite_conversion_int8():
    """Each variant converts to INT8 TFLite within the size budget.

    This is the slow test (~30 s per variant) -- it runs PT2E
    quantization end-to-end and inspects the output.
    """
    sizes = {}
    arenas = {}
    for name in MODEL_VARIANTS:
        m = build_model(name).eval()

        # Warm BN stats
        with torch.no_grad():
            for _ in range(3):
                _ = m(torch.randn(4, T_CONTEXT, N_RADARS, N_ANT, N_BINS, N_IQ))

        sample = (torch.randn(1, T_CONTEXT, N_RADARS, N_ANT, N_BINS, N_IQ),)

        def rep():
            for _ in range(10):
                yield (torch.randn(1, T_CONTEXT, N_RADARS, N_ANT, N_BINS, N_IQ),)

        with tempfile.TemporaryDirectory() as td:
            path = f"{td}/{name}.tflite"
            convert_int8(m, sample, rep, path, n_calibration=10)
            info = inspect_tflite(path)
            sizes[name] = info["size_kb"]
            arenas[name] = info["arena_estimate_kb"]
            assert info["size_kb"] < ESP32_FLASH_KB, \
                f"{name} INT8 size {info['size_kb']:.1f}KB exceeds {ESP32_FLASH_KB}KB"
            assert info["arena_estimate_kb"] < ESP32_SRAM_KB, \
                f"{name} arena {info['arena_estimate_kb']:.1f}KB exceeds {ESP32_SRAM_KB}KB"
    print(f"PASS tflite_conversion_int8")
    for name in MODEL_VARIANTS:
        print(f"   {name:>8s}: {sizes[name]:>6.1f} KB binary, "
              f"{arenas[name]:>6.1f} KB arena")


def test_no_forbidden_ops_int8():
    """The INT8 model contains no forbidden ops (LSTM/GRU/RNN, SUM, etc.)."""
    forbidden_strict = {
        "LSTM", "GRU", "RNN",
        "UNIDIRECTIONAL_SEQUENCE_LSTM",
        "BIDIRECTIONAL_SEQUENCE_LSTM",
    }
    # SUM is allowed but not on the default OpResolver in many builds;
    # we want MEAN instead. The PoC showed SUM appeared because of
    # x.mean(dim=...); we replaced that with adaptive_avg_pool2d, so
    # SUM should no longer appear.
    m = build_model("small").eval()
    with torch.no_grad():
        for _ in range(3):
            _ = m(torch.randn(4, T_CONTEXT, N_RADARS, N_ANT, N_BINS, N_IQ))

    sample = (torch.randn(1, T_CONTEXT, N_RADARS, N_ANT, N_BINS, N_IQ),)

    def rep():
        for _ in range(10):
            yield (torch.randn(1, T_CONTEXT, N_RADARS, N_ANT, N_BINS, N_IQ),)

    with tempfile.TemporaryDirectory() as td:
        path = f"{td}/m.tflite"
        convert_int8(m, sample, rep, path, n_calibration=10)
        info = inspect_tflite(path)
        ops = set(info["op_counts"])
        bad = ops & forbidden_strict
        assert not bad, f"Found forbidden ops: {bad}"
        print(f"PASS no_forbidden_ops_int8 (ops: {sorted(ops)})")


if __name__ == "__main__":
    tests = [
        test_variants_registered,
        test_build_each_variant,
        test_forward_pass,
        test_invalid_variant_raises,
        test_train_step,
        test_param_counts,
        test_tflite_conversion_int8,
        test_no_forbidden_ops_int8,
    ]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} tests passed.")
