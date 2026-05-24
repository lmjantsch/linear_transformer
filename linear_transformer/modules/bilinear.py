from __future__ import annotations

import torch

class BilinearMul(torch.autograd.Function):

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,  # (..., d)
        y: torch.Tensor,  # (..., d)
        x_weight: float = 0.5,
        y_weight: float = 0.5
    ) -> torch.Tensor:  # (..., d)
        orig_dtype = x.dtype
        ctx.save_for_backward(x, y)
        ctx._orig_dtype = orig_dtype
        ctx._x_weight = x_weight
        ctx._y_weight = y_weight
        return (x.float() * y.float()).to(orig_dtype)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_t: torch.Tensor,  # (..., d)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = ctx.saved_tensors
        t_f = grad_t.float()
        grad_x = (ctx._x_weight * t_f * y.float()).to(ctx._orig_dtype)
        grad_y = (ctx._y_weight * t_f * x.float()).to(ctx._orig_dtype)
        return grad_x, grad_y, None, None
    
class BilinearMatmul(torch.autograd.Function):

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,  # (..., d)
        y: torch.Tensor,  # (..., d)
        x_weight: float,
        y_weight: float
    ) -> torch.Tensor:  # (..., d)
        orig_dtype = x.dtype
        ctx.save_for_backward(x, y)
        ctx._orig_dtype = orig_dtype
        ctx._x_weight = x_weight
        ctx._y_weight = y_weight
        return torch.matmul(x.float(), y.float()).to(orig_dtype)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_t: torch.Tensor,  # (..., d)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = ctx.saved_tensors
        t_f = grad_t.float()
        grad_x = (ctx._x_weight * torch.matmul(t_f, y.float().mT)).to(ctx._orig_dtype)
        grad_y = (ctx._y_weight * torch.matmul(x.float().mT, t_f)).to(ctx._orig_dtype)
        return grad_x, grad_y, None, None


# ---------------------------------------------------------------------------
# Functional wrappers
# ---------------------------------------------------------------------------

def mul(x: torch.Tensor, y: torch.Tensor, x_weight: float = 0.5, y_weight: float = 0.5) -> torch.Tensor:
    return torch.mul(x, y)

def matmul(x: torch.Tensor, y: torch.Tensor, x_weight: float = 0.5, y_weight: float = 0.5) -> torch.Tensor:
    return torch.matmul(x, y)


def bilinear_mul(x: torch.Tensor, y: torch.Tensor, x_weight: float = 0.5, y_weight: float = 0.5) -> torch.Tensor:
    return BilinearMul.apply(x, y)


def bilinear_matmul(x: torch.Tensor, y: torch.Tensor, x_weight: float = 0.5, y_weight: float = 0.5) -> torch.Tensor:
    return BilinearMatmul.apply(x, y)


# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------

BILINEAR_FN = {
    # Standard operations
    'mul':    mul,
    'matmul': matmul,
    # Rule 4 — bilinear (LVP uniform splitting)
    'bilinear_mul':    bilinear_mul,
    'bilinear_matmul': bilinear_matmul,
}