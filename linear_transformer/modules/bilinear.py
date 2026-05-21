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
        # save_for_backward only accepts tensors; stash floats on ctx attributes.
        ctx.save_for_backward(x, y)
        ctx._x_weight = x_weight
        ctx._y_weight = y_weight
        ctx._orig_dtype = orig_dtype
        return (x.float() * y.float()).to(orig_dtype)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_t: torch.Tensor,  # (..., d)
    ) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        x, y = ctx.saved_tensors
        x_weight = ctx._x_weight
        y_weight = ctx._y_weight
        t_f = grad_t.float()
        grad_x = (x_weight * t_f * y.float()).to(ctx._orig_dtype)
        grad_y = (y_weight * t_f * x.float()).to(ctx._orig_dtype)
        return grad_x, grad_y, None, None

class BilinearMatmul(torch.autograd.Function):

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,  # (..., d)
        y: torch.Tensor,  # (..., d)
        x_weight: float = 0.5,
        y_weight: float = 0.5,
    ) -> torch.Tensor:  # (..., d)
        orig_dtype = x.dtype
        ctx.save_for_backward(x, y)
        ctx._x_weight = x_weight
        ctx._y_weight = y_weight
        ctx._orig_dtype = orig_dtype
        return torch.matmul(x.float(), y.float()).to(orig_dtype)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_t: torch.Tensor,  # (..., d)
    ) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        x, y = ctx.saved_tensors
        x_weight = ctx._x_weight
        y_weight = ctx._y_weight
        t_f = grad_t.float()
        grad_x = (x_weight * torch.matmul(t_f, y.float().mT)).to(ctx._orig_dtype)
        grad_y = (y_weight * torch.matmul(x.float().mT, t_f)).to(ctx._orig_dtype)
        return grad_x, grad_y, None, None


# ---------------------------------------------------------------------------
# Functional wrappers
# ---------------------------------------------------------------------------

def mul(x: torch.Tensor, y: torch.Tensor, x_weight: float = 0.5, y_weight: float = 0.5) -> torch.Tensor:
    return torch.mul(x, y)

def matmul(x: torch.Tensor, y: torch.Tensor, x_weight: float = 0.5, y_weight: float = 0.5) -> torch.Tensor:
    return torch.matmul(x, y)


def bilinear_mul(x: torch.Tensor, y: torch.Tensor, x_weight: float = 0.5, y_weight: float = 0.5) -> torch.Tensor:
    return BilinearMul.apply(x, y, x_weight, y_weight)


def bilinear_matmul(x: torch.Tensor, y: torch.Tensor, x_weight: float = 0.5, y_weight: float = 0.5) -> torch.Tensor:
    return BilinearMatmul.apply(x, y, x_weight, y_weight)


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