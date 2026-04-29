"""Forward-pass fidelity tests for all supported models.

Tests KL divergence (numerical identity of patched vs unpatched) and
per-module hidden-state divergence across all transformer layers.
Models: gpt2, Qwen/Qwen2.5-0.5B, meta-llama/Llama-3.1-8B, google/gemma-2-2b
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tests.conftest import PROMPTS
from tests.models.utils import rel_l2, get_residual_errors, get_residual_gradient_errors, get_kl_values_from_logits

def test_residual_forward_identity(
    model_config: object,
    org_model_cache: dict,
    identity_model_cache: dict,
    request
) -> None:
    """Patched hidden states must match unpatched at ln2_in (mid) and layer output."""
    num_layers = model_config.num_hidden_layers
    errors = get_residual_errors(identity_model_cache, org_model_cache, num_layers)
    
    max_err = max(max(errors["mid"]), max(errors["out"]))
    passed = max_err < 1e-2

    if not hasattr(pytest, "_model_forward_identity"):
        pytest._model_forward_identity = {}
    pytest._model_forward_identity[request.node.callspec.id] = {"errors": errors, "passed": passed}

    assert passed, (
        f"[{request.node.callspec.id}] max forward divergence {max_err:.2e} exceeds threshold 1e-2"
    )


def test_next_token_kl_identity(
    org_model_cache: dict,
    identity_model_cache: dict,
    request
) -> None:
    """Patched model logits must match unpatched model (KL < 5e-3 per prompt)."""
    kl_values = get_kl_values_from_logits(identity_model_cache, org_model_cache)

    if not hasattr(pytest, "_model_kl_identity"):
        pytest._model_kl_identity = {}
    pytest._model_kl_identity[request.node.callspec.id] = kl_values

    max_kl = max(kl_values)
    assert max_kl < 5e-3, f"[{request.node.callspec.id}] max KL = {max_kl:.2e} exceeds threshold 5e-3"


def test_residual_gradient_identity(
    model_config: object,
    org_model_cache: dict,
    identity_model_cache: dict,
    request
) -> None:
    """Patched hidden states must match unpatched at ln2_in (mid) and layer output."""
    num_layers = model_config.num_hidden_layers
    errors = get_residual_gradient_errors(identity_model_cache, org_model_cache, num_layers)
    
    max_err = max(max(errors["mid_grad"]), max(errors["out_grad"]))
    passed = max_err < 1e-2

    if not hasattr(pytest, "_model_backward_identity"):
        pytest._model_backward_identity = {}
    pytest._model_backward_identity[request.node.callspec.id] = {"errors": errors, "passed": passed}

    assert passed, (
        f"[{request.node.callspec.id}] max forward divergence {max_err:.2e} exceeds threshold 1e-2"
    )


def test_gradient_identity_at_embedding(
    org_model_cache: dict,
    identity_model_cache: dict,
    request
) -> None:
    """Embedding gradient from identity-patched model must match the original model."""
    err = rel_l2(
        org_model_cache['emb_grad'],
        identity_model_cache['emb_grad'],
    )

    if not hasattr(pytest, "_model_emb_grad_identity"):
        pytest._model_emb_grad_identity = {}
    pytest._model_emb_grad_identity[request.node.callspec.id] = err

    assert err < 1e-2, (
        f"[{request.node.callspec.id}] identity-config gradient divergence {err:.2e} exceeds 1e-2"
    )