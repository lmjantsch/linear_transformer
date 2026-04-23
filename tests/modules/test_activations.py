"""Test 4: unit tests for LVP conservation on individual module primitives.

Property: for any LVP-linearized f, (f(x)·t).sum() == (x·x.grad).sum() after f(x).backward(t).
For secant activations f(x) = x·c(x) this is exact; for DTDSoftmax it is approximate.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from linear_transformer.modules.activations import (
    DTDSoftmax,
    SecantGELU,
    SecantGELUTanh,
    SecantReLU,
    SecantSiLU,
    SecantTanh,
)
from .utils import get_conservation_error, register_error

torch.manual_seed(42)
_SHAPE = (2, 8, 64)  # (B, N, d)

# ---------------------------------------------------------------------------
# Zero-preserving activation functions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [SecantGELU, SecantGELUTanh, SecantSiLU, SecantTanh])
def test_secant_activation_conservation(cls: type, device: torch.device) -> None:
    x = torch.randn(_SHAPE, requires_grad=True, device=device)
    t = torch.randn(_SHAPE, device=device)
    out = cls.apply(x)
    out.backward(t)
    err = get_conservation_error(out, [x], t)
    register_error(cls.__name__, err)
    assert err < 1e-5, f"{cls.__name__} conservation error {err:.2e}"


def test_secant_relu_conservation(device: torch.device) -> None:
    x = torch.randn(_SHAPE, requires_grad=True, device=device)
    t = torch.randn(_SHAPE, device=device)
    out = SecantReLU.apply(x, 1e-6)
    out.backward(t)
    err = get_conservation_error(out, [x], t)
    register_error("SecantReLU", err)
    assert err < 1e-5, f"SecantReLU conservation error {err:.2e}"

# ---------------------------------------------------------------------------
# Softmax
# ---------------------------------------------------------------------------

def test_softmax_conservation(device: torch.device) -> None:
    """Softmax is the original softmax function; report the error magnitude."""
    x = torch.randn(_SHAPE, requires_grad=True, device=device)
    t = torch.randn(_SHAPE, device=device)
    out = F.softmax(x, dim=-1, dtype=torch.float32)
    out.backward(t)
    err = get_conservation_error(out, [x], t)
    register_error("Softmax", err)
    assert err < 1.0, f"Softmax conservation error unexpectedly large: {err:.4f}"

def test_dtd_softmax_conservation(device: torch.device) -> None:
    """DTDSoftmax uses an approximation (Deep Taylor Decomp); report the error magnitude."""
    x = torch.randn(_SHAPE, requires_grad=True, device=device)
    t = torch.randn(_SHAPE, device=device)
    out = DTDSoftmax.apply(x, -1, torch.float32)
    out.backward(t)
    err = get_conservation_error(out, [x], t)
    register_error("DTDSoftmax", err)
    assert err < 1.0, f"DTDSoftmax conservation error unexpectedly large: {err:.4f}"

