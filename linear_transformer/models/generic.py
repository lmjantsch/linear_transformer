from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from linear_transformer.modules.linear import linear_sub, linear_add
from linear_transformer.models.utils import baseline_hidden_hook

class CustomLayerNorm(nn.Module):

    def __init__(self, weight: nn.Parameter, bias: nn.Parameter | None, eps: float, normalized_shape: tuple[int, ...],
                 norm_approx: str | None = None) -> None:
        super().__init__()
        self.weight = weight
        self.bias = bias
        self.eps = eps
        self.normalized_shape = normalized_shape
        self.norm_approx = norm_approx

    @classmethod
    def from_module(cls, m: nn.Module, **kwargs: dict | None) -> CustomLayerNorm:
        return cls(
            weight=m.weight,
            bias=m.bias,
            eps=m.eps,
            normalized_shape=tuple(m.normalized_shape),
            norm_approx=kwargs.get('norm_approx', None),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, N, d_model)
        x_f = x.float()
        baseline_hidden = baseline_hidden_hook()
        dynamic_mask = baseline_hidden_hook()

        dims = tuple(range(-len(self.normalized_shape), 0))
        mean = x_f.mean(dim=dims, keepdim=True)
        var = x_f.var(dim=dims, keepdim=True, unbiased=False)
        sigma = (var + self.eps).sqrt()

        if self.norm_approx in ('dynamic_thr', 'dynamic_msk') and (baseline_hidden is not None or dynamic_mask is not None):
            if self.norm_approx == 'dynamic_thr':
                baseline_f = baseline_hidden.to(torch.float32)
                clean_norm = x_f.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                scale = baseline_f.norm(dim=-1, keepdim=True) / clean_norm
                sim = torch.cosine_similarity(x_f, baseline_f, dim=-1).unsqueeze(-1)
                sin2 = (1 - sim.pow(2)).clamp(min=0)
                f_diff = (scale - 1).abs() - ((1 - sim).pow(2) + sin2 * (scale - 1).pow(2)).sqrt()
                dynamic_mask = f_diff >= 0.0
            mean_d = torch.where(dynamic_mask, mean, mean.detach())
            sigma_d = torch.where(dynamic_mask, sigma, sigma.detach())
            out = (x_f - mean_d) / sigma_d * self.weight.float().detach()
            if self.bias is not None:
                out = out + self.bias.detach().float()
            return out.to(x.dtype)

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
            return F.linear(x, self.weight_4_linear.detach(), self.bias.detach())
            
        # original    
        return F.linear(x, self.weight_4_linear, self.bias)
