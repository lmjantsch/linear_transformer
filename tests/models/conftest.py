from __future__ import annotations

import gc
import pytest
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from nnsight import NNsight

from linear_transformer import patch_model_for_lvp
from tests.models.utils import (
    get_arch,
    print_kl_table,
    print_forward_table,
    print_component_table,
    print_conservation_embedding_table,
)

PROMPTS = [
    "Paris is the capital of",
    "The mitochondria is the powerhouse of the",
    "To be or not to be, that is the",
    "The quick brown fox jumps over the",
]

MODEL_IDS = [
    ("gpt2", "gpt2"),
    ("qwen2.5", "Qwen/Qwen2.5-0.5B"),
    ("gemma2", "google/gemma-2-2b"),
    ("llama3.1", "meta-llama/Llama-3.1-8B"),
]

@pytest.fixture(
    scope="session",
    params=[mid for _, mid in MODEL_IDS],
    ids=[name for name, _ in MODEL_IDS],
)
def model_id(request: pytest.FixtureRequest) -> str:
    """Parameterized over 4 models: gpt2, qwen2.5, llama3.1, gemma2."""
    return request.param

@pytest.fixture(scope="session")
def dtype(model_id: str) -> torch.dtype:
    if 'gpt2' in model_id:
        return torch.float
    return torch.bfloat16

@pytest.fixture(scope="session")
def tokenizer(model_id: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if not hasattr(tokenizer, 'pad_token') or not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

@pytest.fixture(scope="session")
def model_config(model_id: str) -> AutoConfig:
    return AutoConfig.from_pretrained(model_id)

@pytest.fixture(scope="session")
def batch_inputs(tokenizer: AutoTokenizer, device: torch.device) -> dict:
    inputs = tokenizer(PROMPTS, return_tensors="pt", padding=True)
    return {k: v.to(device) for k, v in inputs.items()}

@pytest.fixture(scope="session")
def org_model_cache(model_id: str, device: torch.device, dtype: torch.dtype, batch_inputs: dict) -> dict:
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, attn_implementation='eager'
    ).to(device).eval()
    model = NNsight(model)
    arc = get_arch(model_id)
    layers = arc.layers(model)
    cache = {}

    try:
        with torch.no_grad(), model.trace(**batch_inputs):
            for i, layer in enumerate(layers):
                cache[f"{i}.mid"] = getattr(layer, arc.ln2).input.save().cpu()
                cache[f"{i}.out"] = layer.output.save().cpu()
            cache["logits"] = model.lm_head.output[:, -1].float().save().cpu()
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return cache

@pytest.fixture(scope="session")
def patched_model_cache(
    model_id: str, device: torch.device, dtype: torch.dtype, batch_inputs: dict, attn_act_fn: callable
) -> dict:
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, attn_implementation='eager'
    ).to(device).eval()
    model = patch_model_for_lvp(model, attn_act_fn=attn_act_fn)
    arc = get_arch(model_id)
    layers = arc.layers(model)
    num_layers = len(layers)
    cache: dict = {i: {} for i in range(num_layers)}

    try:
        with model.trace(**batch_inputs):
            # Capture residual stream after embed + pos embed — input to first layer's norm.
            # Using ln1 input (not embed output) correctly includes positional embeddings for GPT-2.
            cache['emb'] = getattr(layers[0], arc.ln1).input.save()

            for i, layer in enumerate(layers):
                cache[i]['ln1_in']   = getattr(layer, arc.ln1).input.save()
                cache[i]['ln1_out']  = getattr(layer, arc.ln1).output.save()
                cache[i]['attn_out'] = getattr(layer, arc.attn).output[0].save()
                cache[i]['mid'] = getattr(layer, arc.ln2).input.save()
                cache[i]['ln2_in']   = getattr(layer, arc.ln2).input.save()
                cache[i]['ln2_out']  = getattr(layer, arc.ln2).output.save()
                cache[i]['mlp_out']  = getattr(layer, arc.mlp).output.save()
                cache[i]['out']      = layer.output.save()

            logits = model.lm_head.output[:, -1]          # (B, vocab)
            cache['logits'] = logits.save()
            loss = logits.max(dim=-1).values.sum()         # scalar: sum of argmax logits
            cache['loss'] = loss.save()

            with loss.backward():
                for j in range(num_layers - 1, -1, -1):
                    for k in list(cache[j])[::-1]:
                        if k == 'ln1_in':
                            # isolate grad entering ln1: subtract skip connection grad
                            cache[j][k + '_grad'] = (
                                cache[j][k].grad.save() - cache[j]['mid_grad']
                            ).detach()
                        elif k == 'ln2_in':
                            # isolate grad entering ln2: subtract skip connection grad (= out_grad)
                            cache[j][k + '_grad'] = (
                                cache[j][k].grad.save() - cache[j]['out_grad']
                            ).detach()
                        else:
                            cache[j][k + '_grad'] = cache[j][k].grad.save().detach()

                cache['emb_grad'] = cache['emb'].grad.save()
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Move all tensors to CPU so this model's cache doesn't hold GPU memory during
    # subsequent model fixtures (session-scoped fixtures for all 4 models stay alive
    # simultaneously, so accumulated GPU tensors would otherwise prevent large models from loading).
    for key in list(cache):
        if isinstance(cache[key], dict):
            cache[key] = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                          for k, v in cache[key].items()}
        elif isinstance(cache[key], torch.Tensor):
            cache[key] = cache[key].cpu()
    return cache


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Print summary tables after all tests complete."""
    extended = session.config.getoption("--extended", default=False)

    if hasattr(pytest, "_model_kl_results"):
        print_kl_table(pytest._model_kl_results)

    # Per-model forward divergence: print only for failed models (or all if --extended)
    if hasattr(pytest, "_model_forward_results"):
        for mid, data in pytest._model_forward_results.items():
            if extended or not data["passed"]:
                print_forward_table(mid, data["errors"])

    # Conservation at embedding: always print the compact summary
    if hasattr(pytest, "_model_embedding_ratio"):
        print_conservation_embedding_table(pytest._model_embedding_ratio)

    # Per-model per-module conservation: print only for failed models (or all if --extended)
    if hasattr(pytest, "_model_conservation_results"):
        for mid, data in pytest._model_conservation_results.items():
            if extended or not data["passed"]:
                print_component_table(mid, data["errors"])

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
