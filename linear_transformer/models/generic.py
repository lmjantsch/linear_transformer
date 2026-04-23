from __future__ import annotations

import torch
from torch import nn

class FrozenLayerNorm(nn.Module):

    def __init__(self, weight: nn.Parameter, bias: nn.Parameter | None, eps: float, normalized_shape: tuple[int, ...]) -> None:
        super().__init__()
        self.weight = weight
        self.bias = bias
        self.eps = eps
        self.normalized_shape = normalized_shape

    @classmethod
    def from_module(cls, m: nn.Module, **kwargs: dict | None) -> FrozenLayerNorm:
        return cls(
            weight=m.weight,
            bias=m.bias,
            eps=m.eps,
            normalized_shape=tuple(m.normalized_shape)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, N, d_model)
        x_f = x.float()
        dims = tuple(range(-len(self.normalized_shape), 0))

        mean = x_f.mean(dim=dims, keepdim=True) 
        var = x_f.var(dim=dims, keepdim=True, unbiased=False) 
        sigma = (var + self.eps).sqrt()

        out = (x_f - mean.detach()) / sigma.detach() * self.weight.float().detach()
        if self.bias is not None:
            out = out + self.bias.float().detach()
        return out.to(x.dtype)