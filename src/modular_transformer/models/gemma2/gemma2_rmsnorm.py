from __future__ import annotations

import torch
import torch.nn as nn

from modular_transformer.models.base import ModularModule
from modular_transformer.modules.activations import NORM_FN


class ModularGemma2RMSNorm(ModularModule):

    def __init__(self, weight: nn.Parameter, eps: float, norm_fn: callable) -> None:
        super().__init__()
        self.weight = weight
        self.eps = eps
        self.norm_fn = norm_fn

    @classmethod
    def from_module(cls, m: nn.Module, **kwargs) -> ModularGemma2RMSNorm:
        key = 'frozen_rms_norm' if kwargs.get('norm_approx') == 'frozen' else 'rms_norm'
        return cls(
            weight=m.weight,
            eps=m.eps,
            norm_fn=NORM_FN[key],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.norm_fn(x, self.eps) * (1.0 + self.weight.float())).type_as(x)
