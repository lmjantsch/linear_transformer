from __future__ import annotations

import gc
import time
import pytest
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from nnsight import NNsight

from modular_transformer import patch_model_for_lvp

from tests.models.utils import (
    get_arch,
    cache_forward_and_backward,
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
    # ("gpt2", "gpt2"): [
    #     ("lvp_default", {
    #         "norm_approx": "frozen", "attn_act_fn": "softmax",
    #         "matmul_fn": "bilinear_matmul", "mul_fn": "bilinear_mul",
    #         "mlp_act_fn": "secant_gelu_tanh",
    #     }),
    # ],
    # ("qwen2.5", "Qwen/Qwen2.5-0.5B"): [
    #     ("lvp_default", {
    #         "norm_approx": "frozen", "attn_act_fn": "softmax",
    #         "matmul_fn": "bilinear_matmul", "mul_fn": "bilinear_mul",
    #         "mlp_act_fn": "secant_silu",
    #     }),
    # ],
    ("gemma2", "google/gemma-2-2b"): [
        ("lvp_default", {
            "norm_approx": "frozen", "attn_act_fn": "softmax",
            "matmul_fn": "bilinear_matmul", "mul_fn": "bilinear_mul",
            "mlp_act_fn": "secant_gelu_tanh",
        }),
    ],
    ("llama3.1", "meta-llama/Llama-3.1-8B"): [
        ("lvp_default", {
            "norm_approx": "frozen", "attn_act_fn": "softmax",
            "matmul_fn": "bilinear_matmul", "mul_fn": "bilinear_mul",
            "mlp_act_fn": "secant_silu",
        }),
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
    cache = {}
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, attn_implementation='eager'
    ).to(device).eval()
    try:
        model = NNsight(model)

        cache = cache_forward_and_backward(model_id, model, batch_inputs, cache)
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
    cache = {}
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, attn_implementation='eager'
    ).to(device).eval()
    try:
        model = patch_model_for_lvp(model)

        cache = cache_forward_and_backward(model_id, model, batch_inputs, cache)
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
    cache = {}
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, attn_implementation='eager'
    ).to(device).eval()
    try:
        model = patch_model_for_lvp(model, **lvp_config)

        cache = cache_forward_and_backward(model_id, model, batch_inputs, cache)
    finally:
        del model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return cache


# _test_start: dict[str, float] = {}


# def pytest_runtest_setup(item: pytest.Item) -> None:
#     _test_start[item.nodeid] = time.perf_counter()
#     if torch.cuda.is_available():
#         torch.cuda.reset_peak_memory_stats()


# def pytest_runtest_logreport(report: pytest.TestReport) -> None:
#     if report.when == "teardown":
#         start = _test_start.pop(report.nodeid, None)
#         if start is None:
#             return
#         elapsed = time.perf_counter() - start
#         if torch.cuda.is_available():
#             peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
#             print(f"  [{report.nodeid}] {elapsed:.1f}s | peak CUDA: {peak_mb:.0f} MB", flush=True)
#         else:
#             print(f"  [{report.nodeid}] {elapsed:.1f}s", flush=True)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Print summary tables after all tests complete."""
    extended = session.config.getoption("--extended", default=False)

    # --- Identity tests (no-op patching vs original) ---
    if hasattr(pytest, "_model_kl_identity"):
        print_kl_table(pytest._model_kl_identity)

    if hasattr(pytest, "_model_forward_identity"):
        print_forward_identity_table(pytest._model_forward_identity)
        for mid, data in pytest._model_forward_identity.items():
            if extended or not data["passed"]:
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
            if extended or not data["passed"]:
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
