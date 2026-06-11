from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Annotated, Any

import torch.nn as nn


class ModularModule(nn.Module, ABC):
    """Abstract base for all Modular* wrappers.

    Subclasses must implement :meth:`from_module` to construct the wrapper
    from an existing ``nn.Module`` and optional LVP kwargs.
    """

    @classmethod
    @abstractmethod
    def from_module(cls, m: nn.Module, **kwargs) -> ModularModule:
        """Construct this wrapper from an existing module, sharing its parameters."""
        ...

MODEL_SCOPE = "model"
LAYER_SCOPE = "layer"

ModelCallable = Annotated[Callable[[Any], Any], MODEL_SCOPE]
LayerCallable = Annotated[Callable[[Any], Any], LAYER_SCOPE]

@dataclass(kw_only=True)
class ArchAccessors:
    """Layer accessors for different transformer architectures."""

    # configs
    n_layers: ModelCallable                  # (model) -> int
    n_heads: ModelCallable                   # (model) -> int
    model_dim: ModelCallable                 # (model) -> int
    head_dim: ModelCallable                  # (model) -> int
    uses_rotary_emb: ModelCallable = lambda model: False

    # model modules
    embed: ModelCallable                                                             # (model) -> embed module
    pos_emb: ModelCallable = lambda model: None           # (model) -> position embeding module
    rotary_emb: ModelCallable = lambda model: None    # (model) -> rotary embeding module
    layers: ModelCallable                                                            # (model) -> layer list
    ln: ModelCallable                                                           # (model) -> lm_head module
    lm_head: ModelCallable                                                           # (model) -> lm_head module

    # layer modules
    pre_ln1: LayerCallable                                                           # (layer) -> pre-attention norm module
    attn: LayerCallable                                                              # (layer) -> attention module
    post_ln1: LayerCallable = lambda model:None                         # (layer) -> post-attention norm module

    pre_ln2: LayerCallable                                                           # (layer) -> pre-MLP norm module
    mlp: LayerCallable                                                               # (layer) -> MLP module
    post_ln2: LayerCallable = lambda model: None                        # (layer) -> post-MLP norm module

    # (layer) -> attn submodule
    q_proj: LayerCallable
    k_proj: LayerCallable
    v_proj: LayerCallable
    o_proj: LayerCallable
    attention_interface: LayerCallable

    # (layer) -> mlp submodule
    up_proj: LayerCallable
    gate_proj: LayerCallable = lambda layer: None
    down_proj: LayerCallable