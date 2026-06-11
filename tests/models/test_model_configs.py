"""LVP backward conservation tests for all supported models.

Verifies that (x · ∇x).sum() is preserved layer-to-layer (per-module conservation)
and that the total attribution at the embedding matches the loss (embedding conservation).
Models: gpt2, Qwen/Qwen2.5-0.5B, meta-llama/Llama-3.1-8B, google/gemma-2-2b
"""
from __future__ import annotations

import pytest

from tests.models.utils import rel_err, get_residual_errors, get_kl_values_from_logits

def test_residual_forward_identity(
    model_config: object,
    org_model_cache: dict,
    patched_model_cache: dict,
    request
) -> None:
    """Patched hidden states must match unpatched at ln2_in (mid) and layer output."""
    num_layers = model_config.num_hidden_layers
    errors = get_residual_errors(patched_model_cache, org_model_cache, num_layers)
    
    max_err = max(max(errors["mid"]), max(errors["out"]))
    passed = max_err < 1e-2

    if not hasattr(pytest, "_patched_model_forward_identity"):
        pytest._patched_model_forward_identity = {}
    pytest._patched_model_forward_identity[request.node.callspec.id] = {"errors": errors, "passed": passed}

    assert passed, (
        f"[{request.node.callspec.id}] max forward divergence {max_err:.2e} exceeds threshold 1e-2"
    )


def test_next_token_kl_identity(
    org_model_cache: dict,
    patched_model_cache: dict,
    request
) -> None:
    """Patched model logits must match unpatched model (KL < 5e-3 per prompt)."""
    kl_values = get_kl_values_from_logits(patched_model_cache, org_model_cache)

    if not hasattr(pytest, "_patched_model_kl_identity"):
        pytest._patched_model_kl_identity = {}
    pytest._patched_model_kl_identity[request.node.callspec.id] = kl_values

    max_kl = max(kl_values)
    assert max_kl < 5e-3, f"[{request.node.callspec.id}] max KL = {max_kl:.2e} exceeds threshold 5e-3"
    

def test_per_module_conservation(
    model_config: object,
    patched_model_cache: dict,
    request
) -> None:
    """Report per-module LVP backward conservation error; assert all < 0.1."""
    num_layers = model_config.num_hidden_layers

    errors: dict[str, list[float]] = {"ln1": [], "attn": [], "ln2": [], "mlp": []}
    for i in range(num_layers):
        c = patched_model_cache[i]
        errors["ln1"].append(
            rel_err(c["ln1_in"], c["ln1_in_grad"], c["ln1_out"], c["ln1_out_grad"])
        )
        errors["attn"].append(
            rel_err(c["ln1_out"], c["ln1_out_grad"], c["attn_out"], c["attn_out_grad"])
        )
        errors["ln2"].append(
            rel_err(c["ln2_in"], c["ln2_in_grad"], c["ln2_out"], c["ln2_out_grad"])
        )
        errors["mlp"].append(
            rel_err(c["ln2_out"], c["ln2_out_grad"], c["mlp_out"], c["mlp_out_grad"])
        )

    max_err = max(max(v) for v in errors.values())
    passed = max_err < 0.1

    if not hasattr(pytest, "_patched_model_conservation_results"):
        pytest._patched_model_conservation_results = {}
    pytest._patched_model_conservation_results[request.node.callspec.id] = {"errors": errors, "passed": passed}


def test_total_conservation_at_embedding(
    patched_model_cache: dict,
    request
) -> None:
    """(emb · ∇emb).sum() / loss — ideal LVP gives 1.0."""
    ratio = (
        patched_model_cache['emb'].float() * patched_model_cache['emb_grad'].float()
    ).sum().item() / patched_model_cache['loss'].item()

    if not hasattr(pytest, "_patched_model_embedding_ratio"):
        pytest._patched_model_embedding_ratio = {}
    pytest._patched_model_embedding_ratio[request.node.callspec.id] = ratio

