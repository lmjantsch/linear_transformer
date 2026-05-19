from abc import ABC, abstractmethod

import torch
from torch import nn
from transformers.pytorch_utils import Conv1D

class CustomModule(nn.Module, ABC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    @abstractmethod
    def from_module(cls, m: nn.Module, **kwargs):
        pass

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

class GradientJunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,  # (n, ..., d)
    ) -> torch.Tensor:  # (..., d)
        return x

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_t: torch.Tensor,  # (n+1, ..., d)
    ) -> torch.Tensor:  # (n+1, ..., d)
        grad_t = grad_t.clone()
        for i in range(grad_t.size(0)-1, 0, -1): # .sum() will introduce 1e-2 error (bfloat) due to different addition order
            grad_t[0].add_(grad_t[i])
        return grad_t

  
def hidden_junction_hook(hidden_states: torch.Tensor, n: int) -> torch.Tensor:
    detached_hidden = hidden_states.unsqueeze(0).expand(n, *hidden_states.shape).detach()  # (n, B, N, d_model)
    detached_hidden.requires_grad_(True)

    hidden_states = torch.cat([hidden_states.unsqueeze(0), detached_hidden], dim=0)
    return hidden_states

def gradient_junction_hook(hidden_states: torch.Tensor) -> torch.Tensor:
    return GradientJunction.apply(hidden_states)

def conv1d_to_linear(weight: nn.Parameter, bias: nn.Parameter = None) -> nn.Linear:
    in_f, out_f = weight.shape
    linear = nn.Linear(in_f, out_f, bias=False)
    linear.weight = nn.Parameter(weight.T.contiguous())
    linear.bias = nn.Parameter(bias.clone())
    return linear

def slice_conv1d(conv1d: Conv1D, s: slice) -> nn.Linear:
    return conv1d_to_linear(weight=conv1d.weight[:, s], bias=conv1d.bias[s])
