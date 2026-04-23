from __future__ import annotations

from typing import Callable

import torch.nn as nn

from linear_transformer.models.generic import FrozenLayerNorm
from linear_transformer.models.llama2 import FrozenLlama2RMSNorm, FrozenLlama2SwiGLU, FrozenLlama2Attention
from linear_transformer.models.gemma2 import FrozenGemma2RMSNorm, FrozenGemma2GeGLU, FrozenGemma2Attention
from linear_transformer.models.gpt2 import FrozenGPT2MLP, FrozenGPT2Attention

# Populated below after all wrapper imports
_REGISTRY: dict[type, Callable[[nn.Module], nn.Module]] = {}


def register_lvp_module(
    hf_class: type,
    factory: Callable[[nn.Module], nn.Module],
) -> None:
    """Register a custom HF module class → LVP wrapper factory.

    Call this before patch_model_for_lvp() to support additional model families.

    Example::

        from my_model import MyAttention
        from my_wrappers import MyLVPAttention
        register_lvp_module(MyAttention, MyLVPAttention.from_module)
    """
    _REGISTRY[hf_class] = factory


def _populate_registry() -> None:
    """Lazily import HF classes and fill _REGISTRY.

    Imports are deferred so that missing optional dependencies (e.g. no
    transformers installed) raise only when the registry is first used, not at
    package import time.
    """

    _REGISTRY[nn.LayerNorm] = FrozenLayerNorm.from_module

    try:
        from transformers.models.llama.modeling_llama import (
            LlamaRMSNorm,
            LlamaAttention,
            LlamaMLP,
        )
        _REGISTRY[LlamaRMSNorm] = FrozenLlama2RMSNorm.from_module
        _REGISTRY[LlamaAttention] = FrozenLlama2Attention.from_module
        _REGISTRY[LlamaMLP] = FrozenLlama2SwiGLU.from_module
    except ImportError:
        pass

    try:
        from transformers.models.qwen2.modeling_qwen2 import (
            Qwen2RMSNorm,
            Qwen2Attention,
            Qwen2MLP,
        )
        _REGISTRY[Qwen2RMSNorm] = FrozenLlama2RMSNorm.from_module
        _REGISTRY[Qwen2Attention] = FrozenLlama2Attention.from_module
        _REGISTRY[Qwen2MLP] = FrozenLlama2SwiGLU.from_module
    except ImportError:
        pass

    try:
        from transformers.models.gemma2.modeling_gemma2 import (
            Gemma2RMSNorm,
            Gemma2Attention,
            Gemma2MLP,
        )
        _REGISTRY[Gemma2RMSNorm] = FrozenGemma2RMSNorm.from_module
        _REGISTRY[Gemma2Attention] = FrozenGemma2Attention.from_module
        _REGISTRY[Gemma2MLP] = FrozenGemma2GeGLU.from_module
    except ImportError:
        pass

    try:
        from transformers.models.gpt2.modeling_gpt2 import (
            GPT2Attention,
            GPT2MLP,
        )
        _REGISTRY[GPT2Attention] = FrozenGPT2Attention.from_module
        _REGISTRY[GPT2MLP] = FrozenGPT2MLP.from_module
    except ImportError:
        pass

_populated = False

def get_registry() -> dict[type, Callable[[nn.Module], nn.Module]]:
    global _populated
    if not _populated:
        _populate_registry()
        _populated = True
    return _REGISTRY
