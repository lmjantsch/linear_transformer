from __future__ import annotations

import torch
import torch.nn as nn

from modular_transformer.models.base import ModularModule
from modular_transformer.modules.activations import NORM_FN


class ModularLayerNorm(ModularModule):

    def __init__(
        self,
        weight: nn.Parameter,
        bias: nn.Parameter | None,
        eps: float,
        normalized_shape: tuple[int, ...],
        norm_fn: callable,
    ) -> None:
        super().__init__()
        self.weight = weight
        self.bias = bias
        self.eps = eps
        self.normalized_shape = normalized_shape
        self.norm_fn = norm_fn

    @classmethod
    def from_module(cls, m: nn.Module, **kwargs) -> ModularLayerNorm:
        key = 'frozen_layer_norm' if kwargs.get('norm_approx') == 'frozen' else 'layer_norm'
        return cls(
            weight=m.weight,
            bias=m.bias,
            eps=m.eps,
            normalized_shape=tuple(m.normalized_shape),
            norm_fn=NORM_FN[key],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.norm_fn(x, self.normalized_shape, self.eps) * self.weight.float()
        if self.bias is not None:
            out = out + self.bias.float()
        return out.to(x.dtype)
