"""Test 4: unit tests for LVP conservation on individual module primitives.

Property: for any LVP-linearized f, (f(x)·t).sum() == (x·x.grad).sum() after f(x).backward(t).
For secant activations f(x) = x·c(x) this is exact; for DTDSoftmax it is approximate.
"""
from __future__ import annotations

import pytest
import torch
from linear_transformer.modules.bilinear import BilinearMatmul, BilinearMul
from .utils import get_conservation_error, register_error

torch.manual_seed(42)
_SHAPE = (2, 8, 64)  # (B, N, d)

def test_bilinear_mul_conservation(device: torch.device) -> None:
    a = torch.randn(_SHAPE, requires_grad=True, device=device)
    b = torch.randn(_SHAPE, requires_grad=True, device=device)
    t = torch.randn(_SHAPE, device=device)
    out = BilinearMul.apply(a, b)
    out.backward(t)
    err = get_conservation_error(out, [a, b], t)
    register_error("BilinearMul", err)
    assert err < 1e-5, f"BilinearMul conservation error {err:.2e}"


def test_bilinear_matmul_conservation(device: torch.device) -> None:
    # (B, N, d_in) @ (B, d_in, d_out) → (B, N, d_out)
    x = torch.randn(4, 8, 32, requires_grad=True, device=device)
    y = torch.randn(4, 32, 16, requires_grad=True, device=device)
    t = torch.randn(4, 8, 16, device=device)
    out = BilinearMatmul.apply(x, y)
    out.backward(t)
    err = get_conservation_error(out, [x, y], t)
    register_error("BilinearMatmul", err)
    assert err < 1e-5, f"BilinearMatmul conservation error {err:.2e}"
