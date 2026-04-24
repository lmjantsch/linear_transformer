from __future__ import annotations

import gc
import pytest
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

from linear_transformer import patch_model_for_lvp
from linear_transformer.modules import dtd_softmax, secant_softmax, constant_softmax, pos_ratio_softmax, integrated_softmax, sec_jac_softmax

PROMPTS = [
    "Paris is the capital of",
    "The mitochondria is the powerhouse of the",
    "To be or not to be, that is the",
    "The quick brown fox jumps over the",
]

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device for model and input tensors (e.g. cuda:0, cuda:1, cpu).",
    )
    parser.addoption(
        "--model",
        default=DEFAULT_MODEL,
        help="Model ID for single-model tests (e.g. tests/test_conservation.py).",
    )
    parser.addoption(
        "--attn_act_fn",
        default='dtd_softmax',
        help="softmax implementation to use (e.g. softmax, dtd_softmax)"
    )
    parser.addoption(
        "--matmul_fn",
        default='bilinear_matmul',
        help="Bilinear matmul for QK and AV products (e.g. bilinear_matmul, matmul).",
    )
    parser.addoption(
        "--mul_fn",
        default='bilinear_mul',
        help="Bilinear element-wise mul for gated MLPs (e.g. bilinear_mul, mul).",
    )
    parser.addoption(
        "--mlp_act_fn",
        default=None,
        help="MLP activation key (e.g. secant_silu, secant_gelu_tanh). Defaults to model-specific LVP activation.",
    )
    parser.addoption(
        "--frozen_norm",
        default='true',
        help="Freeze normalisation factors in norms (true/false).",
    )
    parser.addoption(
        "--attn_softcap_fn",
        default='secant_tanh',
        help="Softcap activation for Gemma2 logit softcapping (e.g. secant_tanh, tanh).",
    )
    parser.addoption(
        "--extended",
        action="store_true",
        default=False,
        help="Print extended per-module diagnostics (forward divergence, conservation) in session summary.",
    )

@pytest.fixture(scope="session")
def device(request: pytest.FixtureRequest) -> torch.device:
    return torch.device(request.config.getoption("--device"))

@pytest.fixture(scope="session")
def model_id(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("--model")

@pytest.fixture(scope="session")
def attn_act_fn(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("--attn_act_fn")

@pytest.fixture(scope="session")
def matmul_fn(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("--matmul_fn")

@pytest.fixture(scope="session")
def mul_fn(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("--mul_fn")

@pytest.fixture(scope="session")
def mlp_act_fn(request: pytest.FixtureRequest) -> str | None:
    return request.config.getoption("--mlp_act_fn")

@pytest.fixture(scope="session")
def frozen_norm(request: pytest.FixtureRequest) -> bool:
    return request.config.getoption("--frozen_norm").lower() == 'true'

@pytest.fixture(scope="session")
def attn_softcap_fn(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("--attn_softcap_fn")

@pytest.fixture(scope="session")
def extended(request: pytest.FixtureRequest) -> bool:
    return request.config.getoption("--extended")

def pytest_runtest_teardown(item: pytest.Item) -> None:
    """Clear CUDA cache after each test to prevent OOM between models."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
