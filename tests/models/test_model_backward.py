"""LVP backward conservation tests for all supported models.

Verifies that (x · ∇x).sum() is preserved layer-to-layer (per-module conservation)
and that the total attribution at the embedding matches the loss (embedding conservation).
Models: gpt2, Qwen/Qwen2.5-0.5B, meta-llama/Llama-3.1-8B, google/gemma-2-2b
"""
from __future__ import annotations

import pytest

from tests.models.utils import rel_err


def test_per_module_conservation(
    model_id: str,
    model_config: object,
    patched_model_cache: dict,
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

    if not hasattr(pytest, "_model_conservation_results"):
        pytest._model_conservation_results = {}
    pytest._model_conservation_results[model_id] = {"errors": errors, "passed": passed}

    assert passed, (
        f"[{model_id}] max conservation error {max_err:.2e} exceeds threshold 0.1"
    )


def test_total_conservation_at_embedding(
    model_id: str,
    patched_model_cache: dict,
) -> None:
    """(emb · ∇emb).sum() / loss — ideal LVP gives 1.0."""
    ratio = (
        patched_model_cache['emb'].float() * patched_model_cache['emb_grad'].float()
    ).sum().item() / patched_model_cache['loss'].item()

    if not hasattr(pytest, "_model_embedding_ratio"):
        pytest._model_embedding_ratio = {}
    pytest._model_embedding_ratio[model_id] = ratio

    print(f"\n[{model_id}] (emb · ∇emb).sum() / loss = {ratio:.4f}  (ideal = 1.0)")
    assert abs(ratio - 1.0) < 0.1, (
        f"[{model_id}] embedding conservation ratio {ratio:.4f} deviates from 1.0 by >10%"
    )
