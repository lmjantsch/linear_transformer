from __future__ import annotations

from typing import Callable

import torch.nn as nn

from modular_transformer.models.generic import ModularLayerNorm
from modular_transformer.models.llama2 import ModularLlama2RMSNorm, ModularLlama2MLP, ModularLlama2Attention
from modular_transformer.models.gemma2 import ModularGemma2RMSNorm, ModularGemma2MLP, ModularGemma2Attention
from modular_transformer.models.gpt2 import ModularGPT2MLP, ModularGPT2Attention

# Populated below after all wrapper imports
_REGISTRY: dict[type, Callable[[nn.Module], nn.Module]] = {}


def register_module(
    hf_class: type,
    factory: Callable[[nn.Module], nn.Module],
) -> None:
    _REGISTRY[hf_class] = factory


def _populate_registry() -> None:
    """Lazily import HF classes and fill _REGISTRY.

    Imports are deferred so that missing optional dependencies (e.g. no
    transformers installed) raise only when the registry is first used, not at
    package import time.
    """

    _REGISTRY[nn.LayerNorm] = ModularLayerNorm.from_module

    try:
        from transformers.models.llama.modeling_llama import (
            LlamaRMSNorm,
            LlamaAttention,
            LlamaMLP,
        )
        _REGISTRY[LlamaRMSNorm] = ModularLlama2RMSNorm.from_module
        _REGISTRY[LlamaAttention] = ModularLlama2Attention.from_module
        _REGISTRY[LlamaMLP] = ModularLlama2MLP.from_module
    except ImportError:
        pass

    try:
        from transformers.models.qwen2.modeling_qwen2 import (
            Qwen2RMSNorm,
            Qwen2Attention,
            Qwen2MLP,
        )
        _REGISTRY[Qwen2RMSNorm] = ModularLlama2RMSNorm.from_module
        _REGISTRY[Qwen2Attention] = ModularLlama2Attention.from_module
        _REGISTRY[Qwen2MLP] = ModularLlama2MLP.from_module
    except ImportError:
        pass

    try:
        from transformers.models.gemma2.modeling_gemma2 import (
            Gemma2RMSNorm,
            Gemma2Attention,
            Gemma2MLP,
        )
        _REGISTRY[Gemma2RMSNorm] = ModularGemma2RMSNorm.from_module
        _REGISTRY[Gemma2Attention] = ModularGemma2Attention.from_module
        _REGISTRY[Gemma2MLP] = ModularGemma2MLP.from_module
    except ImportError:
        pass

    try:
        from transformers.models.gpt2.modeling_gpt2 import (
            GPT2Attention,
            GPT2MLP,
        )
        _REGISTRY[GPT2Attention] = ModularGPT2Attention.from_module
        _REGISTRY[GPT2MLP] = ModularGPT2MLP.from_module
    except ImportError:
        pass

_populated = False

def get_registry() -> dict[type, Callable[[nn.Module], nn.Module]]:
    global _populated
    if not _populated:
        _populate_registry()
        _populated = True
    return _REGISTRY
