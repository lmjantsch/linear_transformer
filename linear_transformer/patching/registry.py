from __future__ import annotations

from typing import Callable

import torch.nn as nn

from linear_transformer.models.generic import CustomLayerNorm
from linear_transformer.models.llama2 import CustomLlama2Attention, CustomLlama2RMSNorm, CustomLlamaMLP, CustomLlamaDecoderLayer
from linear_transformer.models.gemma2 import CustomGemma2DecoderLayer, CustomGemma2Attention, CustomGemma2MLP, CustomGemma2RMSNorm
from linear_transformer.models.gpt2 import CustomGPT2MLP, CustomGPT2Attention, CustomGPT2Block

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


def _populate_registry(model_id) -> None:
    """Lazily import HF classes and fill _REGISTRY.

    Imports are deferred so that missing optional dependencies (e.g. no
    transformers installed) raise only when the registry is first used, not at
    package import time.
    """

    _REGISTRY[nn.LayerNorm] = CustomLayerNorm.from_module

    if model_id in ['meta-llama/Llama-3.1-8B']:
        from transformers.models.llama.modeling_llama import (
            LlamaRMSNorm,
            LlamaAttention,
            LlamaMLP,
            LlamaDecoderLayer
        )
        _REGISTRY[LlamaRMSNorm] = CustomLlama2RMSNorm.from_module
        _REGISTRY[LlamaAttention] = CustomLlama2Attention.from_module
        _REGISTRY[LlamaMLP] = CustomLlamaMLP.from_module
        _REGISTRY[LlamaDecoderLayer] = CustomLlamaDecoderLayer.from_module

    if model_id in ['Qwen/Qwen2.5-0.5B']:
        from transformers.models.qwen2.modeling_qwen2 import (
            Qwen2RMSNorm,
            Qwen2Attention,
            Qwen2MLP,
            Qwen2DecoderLayer
        )
        _REGISTRY[Qwen2RMSNorm] = CustomLlama2RMSNorm.from_module
        _REGISTRY[Qwen2Attention] = CustomLlama2Attention.from_module
        _REGISTRY[Qwen2MLP] = CustomLlamaMLP.from_module
        _REGISTRY[Qwen2DecoderLayer] = CustomLlamaDecoderLayer.from_module

    if model_id in ['google/gemma-2-2b']:
        from transformers.models.gemma2.modeling_gemma2 import (
            Gemma2RMSNorm,
            Gemma2Attention,
            Gemma2MLP,
            Gemma2DecoderLayer
        )
        _REGISTRY[Gemma2RMSNorm] = CustomGemma2RMSNorm.from_module
        _REGISTRY[Gemma2Attention] = CustomGemma2Attention.from_module
        _REGISTRY[Gemma2MLP] = CustomGemma2MLP.from_module
        _REGISTRY[Gemma2DecoderLayer] = CustomGemma2DecoderLayer.from_module

    if model_id in ['gpt2']:
        from transformers.models.gpt2.modeling_gpt2 import (
            GPT2Attention,
            GPT2MLP,
            GPT2Block
        )
        _REGISTRY[GPT2Attention] = CustomGPT2Attention.from_module
        _REGISTRY[GPT2MLP] = CustomGPT2MLP.from_module
        _REGISTRY[GPT2Block] = CustomGPT2Block.from_module

_populated_with = None

def get_registry(model_id) -> dict[type, Callable[[nn.Module], nn.Module]]:
    global _populated_with
    if not _populated_with == model_id:
        _populate_registry(model_id)
        _populated_with = model_id
    return _REGISTRY
