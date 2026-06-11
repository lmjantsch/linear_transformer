from __future__ import annotations

import torch
import torch.nn as nn

from modular_transformer.models.base import ModularModule


class ModularLlama2RMSNorm(ModularModule):

    def __init__(self, weight: nn.Parameter, variance_epsilon: float, norm_fn: callable) -> None:
        super().__init__()
        self.weight = weight
        self.variance_epsilon = variance_epsilon
        self.norm_fn = norm_fn

    @classmethod
    def from_module(cls, m: nn.Module, **kwargs) -> ModularLlama2RMSNorm:
        return cls(
            weight=m.weight,
            variance_epsilon=m.variance_epsilon,
            norm_fn=NORM_FN_MAPPING[kwargs.get('norm_approx', None)],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm_fn(self, x)


def llama2_rmsnorm(module: ModularLlama2RMSNorm, x: torch.Tensor) -> torch.Tensor:
    x_f = x.float()
    hidden_states = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + module.variance_epsilon)
    return module.weight * hidden_states.type_as(x)


def llama2_frozen_rmsnorm(module: ModularLlama2RMSNorm, x: torch.Tensor) -> torch.Tensor:
    x_f = x.float()
    hidden_states = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + module.variance_epsilon).detach()
    return module.weight * hidden_states.type_as(x)


NORM_FN_MAPPING = {
    None: llama2_rmsnorm,
    'frozen': llama2_frozen_rmsnorm,
}
