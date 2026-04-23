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
    option = request.config.getoption("--attn_act_fn")
    if option == 'dtd_softmax':
        return dtd_softmax
    if option == 'secant_softmax':
        return secant_softmax
    if option == 'constant_softmax':
        return constant_softmax
    if option == 'pos_ratio_softmax':
        return pos_ratio_softmax
    if option == 'integrated_softmax':
        return integrated_softmax
    if option == 'sec_jac_softmax':
        return sec_jac_softmax
    if option == 'softmax':
        return F.softmax
    raise NotImplementedError(f"Softmax function '{option}' is not implemented.")

@pytest.fixture(scope="session")
def extended(request: pytest.FixtureRequest) -> bool:
    return request.config.getoption("--extended")

def pytest_runtest_teardown(item: pytest.Item) -> None:
    """Clear CUDA cache after each test to prevent OOM between models."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
