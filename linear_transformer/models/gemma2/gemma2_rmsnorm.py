from __future__ import annotations

import torch
import torch.nn as nn

class ModularGemma2RMSNorm(nn.Module):

    def __init__(self, weight: nn.Parameter, eps: float, norm_fn: str | None = None) -> None:
        super().__init__()
        self.weight = weight
        self.eps = eps
        self.norm_fn = norm_fn

    @classmethod
    def from_module(cls, m: nn.Module, **kwargs: dict | None) -> ModularGemma2RMSNorm:
        norm_fn = NORM_FN_MAPPING[kwargs.get('norm_approx', None)]
        return cls(
            weight=m.weight,
            eps=m.eps,
            norm_fn=norm_fn,
        )

    def forward(self, x):
        return self.norm_fn(self, x)
    

    
def gemma2_rmsnorm(module: ModularGemma2RMSNorm, x: torch.Tensor) -> torch.Tensor:
    x_f = x.float()

    output = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + module.eps)
    output = output * (1.0 + module.weight.float())

    return output.type_as(x)


def gemma2_frozen_rmsnorm(module: ModularGemma2RMSNorm, x: torch.Tensor) -> torch.Tensor:
    x_f = x.float()

    output = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + module.eps).detach()
    output = output * (1.0 + module.weight.float())

    return output.type_as(x)



NORM_FN_MAPPING = {
    None: gemma2_rmsnorm,
    'frozen': gemma2_frozen_rmsnorm
}