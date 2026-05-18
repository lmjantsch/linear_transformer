from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from linear_transformer.models.utils import CustomModule

class CustomLayerNorm(CustomModule):

    def __init__(self, weight: nn.Parameter, bias: nn.Parameter | None, eps: float, normalized_shape: tuple[int, ...],
                 norm_approx: str | None = None) -> None:
        super().__init__()
        self.weight = weight
        self.bias = bias
        self.eps = eps
        self.normalized_shape = normalized_shape
        self.norm_approx = norm_approx

    @classmethod
    def from_module(cls, m: nn.Module, **kwargs) -> CustomLayerNorm:
        return cls(
            weight=m.weight,
            bias=m.bias,
            eps=m.eps,
            normalized_shape=tuple(m.normalized_shape),
            norm_approx=kwargs.get('norm_approx', None),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, N, d_model)
        x_f = x.float()

        dims = tuple(range(-len(self.normalized_shape), 0))
        mean = x_f.mean(dim=dims, keepdim=True)
        var = x_f.var(dim=dims, keepdim=True, unbiased=False)
        sigma = (var + self.eps).sqrt()

        # frozen
        if self.norm_approx == 'frozen':
            out = (x_f - mean.detach()) / sigma.detach() * self.weight.float().detach()
            if self.bias is not None:
                out = out + self.bias.detach().float()
            return out.to(x.dtype)

        # original
        out = (x_f - mean) / sigma * self.weight.float()
        if self.bias is not None:
            out = out + self.bias.float()
        return out.to(x.dtype)
