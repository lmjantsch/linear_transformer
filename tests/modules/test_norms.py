"""Unit tests for norm non-linearities in activations.py.

Tests:
- forward identity: frozen and standard produce the same output
- frozen detaches the normalization factor (grad does not flow through it)
- standard norm allows grad to flow through the normalization factor
"""
from __future__ import annotations

import pytest
import torch

from modular_transformer.modules.activations import (
    frozen_layer_norm,
    frozen_rms_norm,
    layer_norm,
    rms_norm,
)

torch.manual_seed(42)
_EPS = 1e-5
_SHAPE = (2, 8, 64)  # (B, N, d)


# ---------------------------------------------------------------------------
# Forward identity: frozen == standard in the forward pass
# ---------------------------------------------------------------------------

def test_rms_norm_frozen_forward_identity(device: torch.device) -> None:
    x = torch.randn(_SHAPE, device=device)
    assert torch.allclose(rms_norm(x, _EPS), frozen_rms_norm(x, _EPS), atol=1e-6)


def test_layer_norm_frozen_forward_identity(device: torch.device) -> None:
    x = torch.randn(_SHAPE, device=device)
    assert torch.allclose(
        layer_norm(x, (_SHAPE[-1],), _EPS),
        frozen_layer_norm(x, (_SHAPE[-1],), _EPS),
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# Output dtype: always float32 regardless of input dtype
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fn,extra", [
    (rms_norm,        (_EPS,)),
    (frozen_rms_norm, (_EPS,)),
    (layer_norm,        ((_SHAPE[-1],), _EPS)),
    (frozen_layer_norm, ((_SHAPE[-1],), _EPS)),
])
def test_norm_output_is_float32(fn, extra, device: torch.device) -> None:
    x = torch.randn(_SHAPE, device=device, dtype=torch.bfloat16)
    out = fn(x, *extra)
    assert out.dtype == torch.float32, f"{fn.__name__} output dtype should be float32"


# ---------------------------------------------------------------------------
# Gradient flow: standard passes grad through factor; frozen blocks it
# ---------------------------------------------------------------------------

def test_rms_norm_grad_flows(device: torch.device) -> None:
    x = torch.randn(_SHAPE, requires_grad=True, device=device)
    t = torch.randn(_SHAPE, device=device)
    out = rms_norm(x, _EPS)
    out.backward(t)
    assert x.grad is not None
    assert not torch.all(x.grad == 0), "rms_norm should have non-zero grad through factor"


def test_frozen_rms_norm_grad_matches_linear(device: torch.device) -> None:
    """With frozen factor, backward is linear in x: grad ∝ t * factor (detached)."""
    x = torch.randn(_SHAPE, requires_grad=True, device=device)
    t = torch.randn(_SHAPE, device=device)
    out = frozen_rms_norm(x, _EPS)

    # Compute the detached scale factor for reference
    with torch.no_grad():
        x_f = x.float()
        scale = torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + _EPS)  # detached reference

    out.backward(t)
    expected_grad = (t.float() * scale).to(x.dtype)
    # The grad of x should be proportional to t * scale (linear rule)
    assert torch.allclose(x.grad.float(), expected_grad.float(), atol=1e-5), (
        "frozen_rms_norm backward should equal t * detached_scale"
    )


def test_layer_norm_grad_flows(device: torch.device) -> None:
    x = torch.randn(_SHAPE, requires_grad=True, device=device)
    t = torch.randn(_SHAPE, device=device)
    out = layer_norm(x, (_SHAPE[-1],), _EPS)
    out.backward(t)
    assert x.grad is not None
    assert not torch.all(x.grad == 0), "layer_norm should have non-zero grad through stats"


def test_frozen_layer_norm_grad_differs_from_standard(device: torch.device) -> None:
    """Frozen and standard layer norm should produce different gradients."""
    x = torch.randn(_SHAPE, requires_grad=True, device=device)
    x_frozen = x.detach().requires_grad_(True)
    t = torch.randn(_SHAPE, device=device)

    layer_norm(x, (_SHAPE[-1],), _EPS).backward(t)
    frozen_layer_norm(x_frozen, (_SHAPE[-1],), _EPS).backward(t)

    assert not torch.allclose(x.grad.float(), x_frozen.grad.float(), atol=1e-6), (
        "frozen and standard layer_norm should differ in backward"
    )
