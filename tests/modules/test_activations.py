"""Test 4: unit tests for LVP conservation on individual module primitives.

Property: for any LVP-linearized f, (f(x)·t).sum() == (x·x.grad).sum() after f(x).backward(t).
For secant activations f(x) = x·c(x) this is exact; for DTDSoftmax it is approximate.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from linear_transformer.modules.activations import (
    IntegratedSoftmax,
    SecantGELUTanh,
    SecantJacobianSoftmax,
    SecantSiLU,
)
from .utils import get_conservation_error, register_error

torch.manual_seed(42)
_SHAPE = (2, 8, 64)  # (B, N, d)

# ---------------------------------------------------------------------------
# Zero-preserving activation functions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [SecantGELUTanh, SecantSiLU])
def test_secant_activation_conservation(cls: type, device: torch.device) -> None:
    x = torch.randn(_SHAPE, requires_grad=True, device=device)
    t = torch.randn(_SHAPE, device=device)
    out = cls.apply(x)
    out.backward(t)
    err = get_conservation_error(out, [x], t)
    register_error(cls.__name__, err)
    assert err < 1e-5, f"{cls.__name__} conservation error {err:.2e}"

# ---------------------------------------------------------------------------
# Softmax — conservation (pre-existing; seed-state-sensitive)
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

# ---------------------------------------------------------------------------
# Softmax — all variants share the same forward (new tests; each sets its own seed)
# ---------------------------------------------------------------------------

_SOFTMAX_VARIANTS = [
    IntegratedSoftmax,
    SecantJacobianSoftmax
]


@pytest.mark.parametrize("cls", _SOFTMAX_VARIANTS)
def test_softmax_variant_forward(
    cls: type, shape: tuple[int, int, int], device: torch.device
) -> None:
    """Every LVP softmax variant produces the same output as F.softmax."""
    torch.manual_seed(0)
    x = torch.randn(shape, device=device)
    expected = F.softmax(x, dim=-1, dtype=torch.float32)
    out = cls.apply(x, -1, torch.float32)
    assert torch.allclose(out.float(), expected.float(), atol=1e-5), (
        f"{cls.__name__} forward does not match F.softmax"
    )


@pytest.mark.parametrize("cls", _SOFTMAX_VARIANTS)
def test_softmax_variant_backward_shape(
    cls: type, shape: tuple[int, int, int], device: torch.device
) -> None:
    """Backward produces a gradient with the same shape as the input."""
    torch.manual_seed(0)
    x = torch.randn(shape, requires_grad=True, device=device)
    t = torch.randn(shape, device=device)
    out = cls.apply(x, -1, torch.float32)
    out.backward(t)
    assert x.grad is not None
    assert x.grad.shape == x.shape


@pytest.mark.parametrize("cls", [IntegratedSoftmax, SecantJacobianSoftmax])
def test_softmax_approximate_conservation(
    cls: type, shape: tuple[int, int, int], device: torch.device
) -> None:
    """Principled approximate variants; report worst-case error and check it stays below 1.0."""
    torch.manual_seed(0)
    x = torch.randn(shape, requires_grad=True, device=device)
    t = torch.randn(shape, device=device)
    out = cls.apply(x, -1, torch.float32)
    out.backward(t)
    err = get_conservation_error(out, [x], t)
    register_error(cls.__name__, err)
    assert err < 1.0, f"{cls.__name__} conservation error unexpectedly large: {err:.4f}"
