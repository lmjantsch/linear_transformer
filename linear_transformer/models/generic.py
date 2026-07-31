from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from linear_transformer.models.utils import baseline_hidden_hook


class IGLayerNorm(torch.autograd.Function):
    """Integrated-gradient backward through LayerNorm normalization statistics."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,         # (..., *normalized_shape)
        baseline_hidden: torch.Tensor,
        eps: float,
        dims: tuple,
    ) -> torch.Tensor:           # (..., *normalized_shape), weight/bias not applied
        ctx.save_for_backward(x, baseline_hidden)
        ctx._eps = eps
        ctx._dims = dims
        x_f = x.float()
        mean = x_f.mean(dim=dims, keepdim=True)
        var = x_f.var(dim=dims, keepdim=True, unbiased=False)
        return (x_f - mean.detach()) / (var + eps).sqrt().detach()

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_t: torch.Tensor,    # (..., *normalized_shape)
    ) -> tuple:
        (x, baseline_hidden) = ctx.saved_tensors
        x_f, bh_f = x.float(), baseline_hidden.float()
        eps = ctx._eps
        dims = ctx._dims
        grad_f = grad_t.float()

        delta_x = x_f - bh_f
        expected_grad = torch.zeros_like(x_f)
        steps = 32

        for i in range(steps):
            alpha = (i + 0.5) / steps
            x_a = bh_f + alpha * delta_x
            mean_a = x_a.mean(dim=dims, keepdim=True)
            sigma_a = (x_a.var(dim=dims, keepdim=True, unbiased=False) + eps).sqrt()
            y_a = (x_a - mean_a) / sigma_a
            # J(x_a)^T @ t = (1/sigma) * [t - mean(t) - y * mean(y*t)]
            jac_t = (grad_f - grad_f.mean(dim=dims, keepdim=True) - y_a * (y_a * grad_f).mean(dim=dims, keepdim=True)) / sigma_a
            expected_grad += jac_t

        expected_grad = expected_grad / steps
        return expected_grad.to(x.dtype), None, None, None


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

        # ig
        if self.norm_approx == 'ig' and baseline_hidden is not None:
            output = IGLayerNorm.apply(x, baseline_hidden, self.eps, dims)
            output = output * self.weight.float()
            if self.bias is not None:
                output = output + self.bias.float()
            return output.to(x.dtype)

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
