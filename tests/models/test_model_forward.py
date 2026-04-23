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
from tests.models.utils import rel_l2, print_forward_table


def test_per_module_forward_divergence(
    model_id: str,
    model_config: object,
    org_model_cache: dict,
    patched_model_cache: dict,
) -> None:
    """Patched hidden states must match unpatched at ln2_in (mid) and layer output."""
    num_layers = model_config.num_hidden_layers
    errors: dict[str, list[float]] = {"mid": [], "out": []}

    for i in range(num_layers):
        errors["mid"].append(
            rel_l2(org_model_cache[f"{i}.mid"], patched_model_cache[i]["ln2_in"])
        )
        errors["out"].append(
            rel_l2(org_model_cache[f"{i}.out"], patched_model_cache[i]["out"])
        )

    max_err = max(max(errors["mid"]), max(errors["out"]))
    passed = max_err < 1e-2

    if not hasattr(pytest, "_model_forward_results"):
        pytest._model_forward_results = {}
    pytest._model_forward_results[model_id] = {"errors": errors, "passed": passed}

    assert passed, (
        f"[{model_id}] max forward divergence {max_err:.2e} exceeds threshold 1e-2"
    )


def test_next_token_kl_divergence(
    model_id: str,
    org_model_cache: dict,
    patched_model_cache: dict,
) -> None:
    """Patched model logits must match unpatched model (KL < 5e-3 per prompt)."""
    kl_values: list[float] = []
    for b, prompt in enumerate(PROMPTS):
        p = F.softmax(org_model_cache['logits'][b, :], dim=-1)
        q = F.softmax(patched_model_cache['logits'][b, :], dim=-1)
        kl = (p * (p.log() - q.log())).sum().item()
        kl_values.append(kl)
        print(f"  [{model_id}] [{prompt[:32]:<32s}] KL = {kl:.2e}")

    if not hasattr(pytest, "_model_kl_results"):
        pytest._model_kl_results = {}
    pytest._model_kl_results[model_id] = kl_values

    max_kl = max(kl_values)
    assert max_kl < 5e-3, f"[{model_id}] max KL = {max_kl:.2e} exceeds threshold 5e-3"
