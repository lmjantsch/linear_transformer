from __future__ import annotations

import torch
import torch.nn as nn

from modular_transformer.models.base import ModularModule
from modular_transformer.modules.activations import NORM_FN


class ModularLlama2RMSNorm(ModularModule):

    def __init__(self, weight: nn.Parameter, variance_epsilon: float, norm_fn: callable) -> None:
        super().__init__()
        self.weight = weight
        self.variance_epsilon = variance_epsilon
        self.norm_fn = norm_fn

    @classmethod
    def from_module(cls, m: nn.Module, **kwargs) -> ModularLlama2RMSNorm:
        key = 'frozen_rms_norm' if kwargs.get('norm_approx') == 'frozen' else 'rms_norm'
        return cls(
            weight=m.weight,
            variance_epsilon=m.variance_epsilon,
            norm_fn=NORM_FN[key],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * self.norm_fn(x, self.variance_epsilon).type_as(x)
