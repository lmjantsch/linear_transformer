from __future__ import annotations

import torch
import torch.nn as nn

class ModularLayerNorm(nn.Module):

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
        return cls(
            weight=m.weight,
            bias=m.bias,
            eps=m.eps,
            normalized_shape=tuple(m.normalized_shape),
            norm_fn=NORM_FN_MAPPING[kwargs.get('norm_approx', None)],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm_fn(self, x)


def layernorm(module: ModularLayerNorm, x: torch.Tensor) -> torch.Tensor:
    x_f = x.float()
    dims = tuple(range(-len(module.normalized_shape), 0))
    mean = x_f.mean(dim=dims, keepdim=True)
    var = x_f.var(dim=dims, keepdim=True, unbiased=False)
    out = (x_f - mean) / (var + module.eps).sqrt()
    out = out *module.weight.float()
    if module.bias is not None:
        out = out + module.bias.float()
    return out.to(x.dtype)


def frozen_layernorm(module: ModularLayerNorm, x: torch.Tensor) -> torch.Tensor:
    x_f = x.float()
    dims = tuple(range(-len(module.normalized_shape), 0))
    mean = x_f.mean(dim=dims, keepdim=True).detach()
    var = x_f.var(dim=dims, keepdim=True, unbiased=False).detach()
    out = (x_f - mean) / (var + module.eps).sqrt() 
    out = out * module.weight.float()
    if module.bias is not None:
        out = out + module.bias.float()
    return out.to(x.dtype)


NORM_FN_MAPPING = {
    None: layernorm,
    'frozen': frozen_layernorm,
}
