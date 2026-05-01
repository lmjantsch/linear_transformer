from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from linear_transformer.modules.linear import linear_sub, linear_add

class CustomLayerNorm(nn.Module):

    def __init__(self, weight: nn.Parameter, bias: nn.Parameter | None, eps: float, normalized_shape: tuple[int, ...], is_linear: bool = False) -> None:
        super().__init__()
        self.weight = weight
        self.bias = bias
        self.eps = eps
        self.normalized_shape = normalized_shape
        self.is_linear = is_linear

    @classmethod
    def from_module(cls, m: nn.Module, **kwargs: dict | None) -> CustomLayerNorm:
        return cls(
            weight=m.weight,
            bias=m.bias,
            eps=m.eps,
            normalized_shape=tuple(m.normalized_shape),
            is_linear=kwargs.get('frozen_norm', False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, N, d_model)
        x_f = x.float()
        dims = tuple(range(-len(self.normalized_shape), 0))

        mean = x_f.mean(dim=dims, keepdim=True) 
        var = x_f.var(dim=dims, keepdim=True, unbiased=False) 
        sigma = (var + self.eps).sqrt()

        # linearized
        if self.is_linear:
            mean_cent_x_f = linear_sub(x_f, mean.detach(), self.eps)
            out = mean_cent_x_f / sigma.detach() * self.weight.float().detach()
            if self.bias is not None:
                out = linear_add(out, self.bias.float().detach(), self.eps)
            return out.to(x.dtype)

        # original
        out = (x_f - mean) / sigma * self.weight.float()
        if self.bias is not None:
            out = out + self.bias.float()
        return out.to(x.dtype)
    

class CustomLinear(nn.Module):

    def __init__(self, weight: nn.Parameter, bias: nn.Parameter | None, is_linear: bool = False, reversed: bool = False):
        super().__init__()
        self.weight = weight
        self.bias = bias
        self.is_linear = is_linear
        self.reversed = reversed

    @property
    def weight_4_linear(self):
        if self.reversed:
            return self.weight.T
        return self.weight

    @classmethod
    def from_module(cls, m: nn.Module, **kwargs: dict) -> 'CustomLinear':
        reversed = False
        if isinstance(m, nn.Conv1d):
            reversed = True
        return cls(
            weight = m.weight,
            bias = m.bias,
            is_linear = kwargs.get('frozen_linear', False),
            reversed = reversed
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # linearized
        if self.is_linear:
            out = F.linear(x, self.weight_4_linear.detach())
            if self.bias is not None:
                out = linear_add(out, self.bias.detach(), 1e-8)
            return out
            
        # original    
        return F.linear(x, self.weight_4_linear, self.bias)
