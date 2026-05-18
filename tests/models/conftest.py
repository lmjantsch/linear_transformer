from __future__ import annotations

import gc
import time
import pytest
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from nnsight import NNsight

from linear_transformer import patch_model_for_lvp

from tests.models.utils import (
    cache_org_model,
    cache_patched_model,
    print_kl_table,
    print_forward_table,
    print_forward_identity_table,
    print_backward_identity_table,
    print_emb_grad_identity_table,
    print_component_table,
    print_conservation_embedding_table,
    print_conservation_summary_table,
)

PROMPTS = [
    "Paris is the capital of",
    "The mitochondria is the powerhouse of the",
    "To be or not to be, that is the",
    "The quick brown fox jumps over the",
]

MODEL_CONFIG_MAPPING: dict[str, list[tuple[str, dict]]] = {
    ("gpt2", "gpt2"): [
        # ("fr_norm", {
        #     "frozen_norm": True
        # }),
        # ("secant_mlp", {
        #     "mlp_act_fn": "secant_gelu_tanh"
        # }),
        # ("bilinear", {
        #     "matmul_fn": "bilinear_matmul", "mul_fn": "bilinear_mul"
        # }),
        # ("integrated_softmax", {
        #     "attn_act_fn": "integrated_softmax",
        # }),
        # ("sec_jac_softmax", {
        #     "attn_act_fn": "sec_jac_softmax",
        # }),
    ],
    ("qwen2.5", "Qwen/Qwen2.5-0.5B"): [
        # ("fr_norm", {
        #     "frozen_norm": True
        # }),
        # ("secant_mlp", {
        #     "mlp_act_fn": "secant_silu"
        # }),
        # ("bilinear", {
        #     "matmul_fn": "bilinear_matmul", "mul_fn": "bilinear_mul"
        # }),
        # ("integrated_softmax", {
        #     "attn_act_fn": "integrated_softmax",
        # }),
        # ("sec_jac_softmax", {
        #     "attn_act_fn": "sec_jac_softmax",
        # }),
    ],
    ("gemma2", "google/gemma-2-2b"): [
        # ("fr_norm", {
        #     "frozen_norm": True
        # }),
        # ("secant_mlp", {
        #     "mlp_act_fn": "secant_gelu_tanh"
        # }),
        # ("bilinear", {
        #     "matmul_fn": "bilinear_matmul", "mul_fn": "bilinear_mul"
        # }),
        # ("integrated_softmax", {
        #     "attn_act_fn": "integrated_softmax",
        # }),
        # ("sec_jac_softmax", {
        #     "attn_act_fn": "sec_jac_softmax",
        # }),
    ],
    ("llama3.1", "meta-llama/Llama-3.1-8B"): [
        # ("fr_norm", {
        #     "frozen_norm": True
        # }),
        # ("secant_mlp", {
        #     "mlp_act_fn": "secant_silu"
        # }),
        # ("bilinear", {
        #     "matmul_fn": "bilinear_matmul", "mul_fn": "bilinear_mul"
        # }),
        # ("integrated_softmax", {
        #     "attn_act_fn": "integrated_softmax",
        # }),
        # ("sec_jac_softmax", {
        #     "attn_act_fn": "sec_jac_softmax",
        # }),
    ],
}

def pytest_generate_tests(metafunc):
    if "model_id" in metafunc.fixturenames and "lvp_config" in metafunc.fixturenames:
        params = []
        ids = []
        for model, configs in MODEL_CONFIG_MAPPING.items():
            model_name, model_id = model
            for cfg in configs:
                cfg_name, cfg = cfg
                params.append((model_id, cfg))
                ids.append(f"{model_name}-{cfg_name}")
        
        metafunc.parametrize("model_id, lvp_config", params, ids=ids, scope="session")
    
    elif "model_id" in metafunc.fixturenames:
        model_names, model_ids = list(zip(*MODEL_CONFIG_MAPPING.keys()))
        metafunc.parametrize("model_id", model_ids, ids=model_names, scope="session")

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
    try:
        model = NNsight(model)
        cache = cache_org_model(model_id, model, batch_inputs)
    finally:
        del model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return cache

@pytest.fixture(scope="session")
def identity_model_cache(
    model_id: str, device: torch.device, dtype: torch.dtype, batch_inputs: dict
) -> dict:
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, attn_implementation='eager'
    ).to(device).eval()
    try:
        model = patch_model_for_lvp(model, model_id)
        cache = cache_patched_model(model_id, model, batch_inputs)
    finally:
        del model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return cache

@pytest.fixture(scope="session")
def patched_model_cache(
    model_id: str, lvp_config: dict, device: torch.device, dtype: torch.dtype, batch_inputs: dict
) -> dict:
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, attn_implementation='eager'
    ).to(device).eval()
    try:
        model = patch_model_for_lvp(model, model_id, **lvp_config)
        cache = cache_patched_model(model_id, model, batch_inputs)
    finally:
        del model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return cache


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Print summary tables after all tests complete."""
    extended = session.config.getoption("--extended", default=False)

    # --- Identity tests (no-op patching vs original) ---
    if hasattr(pytest, "_model_kl_identity"):
        print_kl_table(pytest._model_kl_identity)

    if hasattr(pytest, "_model_forward_identity"):
        print_forward_identity_table(pytest._model_forward_identity)
        for mid, data in pytest._model_forward_identity.items():
            if not data["passed"]:
                print_forward_table(mid, data["errors"])

    if hasattr(pytest, "_model_backward_identity"):
        print_backward_identity_table(pytest._model_backward_identity)

    if hasattr(pytest, "_model_emb_grad_identity"):
        print_emb_grad_identity_table(pytest._model_emb_grad_identity)

    # --- LVP config tests (conservation) ---
    if hasattr(pytest, "_lvp_model_kl_identity"):
        print_kl_table(pytest._lvp_model_kl_identity)

    if hasattr(pytest, "_lvp_model_forward_identity"):
        print_forward_identity_table(pytest._lvp_model_forward_identity)
        for mid, data in pytest._lvp_model_forward_identity.items():
            if not data["passed"]:
                print_forward_table(mid, data["errors"])

    if hasattr(pytest, "_lvp_model_embedding_ratio"):
        print_conservation_embedding_table(pytest._lvp_model_embedding_ratio)

    if hasattr(pytest, "_lvp_model_conservation_results"):
        print_conservation_summary_table(pytest._lvp_model_conservation_results)
        if extended:
            for mid, data in pytest._lvp_model_conservation_results.items():
                print_component_table(mid, data["errors"])

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
