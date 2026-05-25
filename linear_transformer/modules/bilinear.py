from __future__ import annotations

import torch
import math

def fwd_context_hook() -> dict:
    return {}

def get_slerp_mid(v: torch.Tensor, v_alt: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    # clamp min to prevent division by zero if vectors are perfectly anti-parallel (pi)
    cos_half_theta = torch.cos(0.5 * theta).clamp(min=1e-5)
    scale = 1.0 / (2.0 * cos_half_theta)
    # print(v.shape, theta.shape)
    out = (v + v_alt).float() * scale
    return out.to(v.dtype)

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
    

class DynamicBilinearMul(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,       # Current state (e.g., clean)
        y: torch.Tensor,       # Current state (e.g., clean)
        x_base: torch.Tensor,
        y_base: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x_mid = 0.5 * (x + x_base)
        y_mid = 0.5 * (y + y_base)
    
        ctx.save_for_backward(x_mid, y_mid)
        ctx._orig_dtype = orig_dtype
        
        return (x.float() * y.float()).to(orig_dtype)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_t: torch.Tensor, 
    ) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        x_mid, y_mid = ctx.saved_tensors
        t_f = grad_t.float()
        
        grad_x = (t_f * y_mid.float()).to(ctx._orig_dtype)
        grad_y = (t_f * x_mid.float()).to(ctx._orig_dtype)

        return grad_x, grad_y, None, None
    
class SlerpBilinearMul(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,       # Current state (e.g., clean)
        y: torch.Tensor,       # Current state (e.g., clean)
        x_base: torch.Tensor,
        y_base: torch.Tensor,
        theta_x: torch.Tensor,
        theta_y: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x_mid = get_slerp_mid(x, x_base, theta_x)
        y_mid = get_slerp_mid(y, y_base, theta_y)
    
        ctx.save_for_backward(x_mid, y_mid)
        ctx._orig_dtype = orig_dtype
        
        return (x.float() * y.float()).to(orig_dtype)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_t: torch.Tensor, 
    ) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        x_mid, y_mid = ctx.saved_tensors
        t_f = grad_t.float()
        
        grad_x = (t_f * y_mid.float()).to(ctx._orig_dtype)
        grad_y = (t_f * x_mid.float()).to(ctx._orig_dtype)

        return grad_x, grad_y, None, None, None, None
    
class BilinearMatmul(torch.autograd.Function):

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
    
class DynamicBilinearMatmul(torch.autograd.Function):

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,       # (..., n, m) Current state (e.g., clean)
        y: torch.Tensor,       # (..., m, p) Current state (e.g., clean)
        x_base: torch.Tensor,
        y_base: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x_mid = 0.5 * (x + x_base)
        y_mid = 0.5 * (y + y_base)
        
        ctx.save_for_backward(x_mid, y_mid)
        ctx._orig_dtype = orig_dtype
        
        # The forward pass uses the raw x and y. 
        # The model's behavior and logits remain 100% unchanged.
        return torch.matmul(x.float(), y.float()).to(orig_dtype)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_t: torch.Tensor,  # (..., n, p)
    ) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        x_mid, y_mid = ctx.saved_tensors
        t_f = grad_t.float()
        
        grad_x = torch.matmul(t_f, y_mid.float().mT).to(ctx._orig_dtype)
        grad_y = torch.matmul(x_mid.float().mT, t_f).to(ctx._orig_dtype)
        
        return grad_x, grad_y, None, None
    
class SlerpBilinearMatmul(torch.autograd.Function):

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,       # (..., n, m) Current state (e.g., clean)
        y: torch.Tensor,       # (..., m, p) Current state (e.g., clean)
        x_base: torch.Tensor,   # (..., n, m) Counterfactual state (e.g., corrupt)
        y_base: torch.Tensor,    # (..., m, p) Counterfactual state (e.g., corrupt)
        theta_x: torch.Tensor,
        theta_y: torch.Tensor,
        is_av_matmul: bool
    ) -> torch.Tensor:
        orig_dtype = x.dtype
    
        if is_av_matmul:
            x_mid = 0.5 * (x + x_base)
        else:
            x_mid = get_slerp_mid(x, x_base, theta_x)
        y_mid = get_slerp_mid(y, y_base, theta_y)
        
        ctx.save_for_backward(x_mid, y_mid)
        ctx._orig_dtype = orig_dtype
        
        # The forward pass uses the raw x and y. 
        # The model's behavior and logits remain 100% unchanged.
        return torch.matmul(x.float(), y.float()).to(orig_dtype)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_t: torch.Tensor,  # (..., n, p)
    ) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        x_mid, y_mid = ctx.saved_tensors
        t_f = grad_t.float()
        
        grad_x = torch.matmul(t_f, y_mid.float().mT).to(ctx._orig_dtype)
        grad_y = torch.matmul(x_mid.float().mT, t_f).to(ctx._orig_dtype)
        
        return grad_x, grad_y, None, None, None, None, None


# ---------------------------------------------------------------------------
# Functional wrappers
# ---------------------------------------------------------------------------

def mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    fwd_context = fwd_context_hook()
    return torch.mul(x, y)

def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    fwd_context = fwd_context_hook()
    return torch.matmul(x, y)

def bilinear_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    fwd_context = fwd_context_hook()
    return BilinearMul.apply(x, y, fwd_context.get("x_weight", 0.5), fwd_context.get("y_weight", 0.5))

def ig_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    fwd_context = fwd_context_hook()
    return DynamicBilinearMul.apply(x, y, fwd_context.get("x_base", torch.zeros_like(x)), fwd_context.get("y_base", torch.zeros_like(y)))

def slerp_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    fwd_context = fwd_context_hook()
    return SlerpBilinearMul.apply(x, y, fwd_context.get("x_base", torch.zeros_like(x)), fwd_context.get("y_base", torch.zeros_like(y)),
                                  fwd_context.get("theta", torch.full_like(x, 2 * math.pi / 3)), fwd_context.get("theta", torch.full_like(y, 2 * math.pi / 3)))

def bilinear_matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    fwd_context = fwd_context_hook()
    return BilinearMatmul.apply(x, y, fwd_context.get("x_weight", 0.5), fwd_context.get("y_weight", 0.5))

def ig_matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    fwd_context = fwd_context_hook()
    return DynamicBilinearMatmul.apply(x, y, fwd_context.get("x_base", torch.zeros_like(x)), fwd_context.get("y_base", torch.zeros_like(y)))

def slerp_matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    fwd_context = fwd_context_hook()
    return SlerpBilinearMatmul.apply(x, y, fwd_context.get("x_base", torch.zeros_like(x)), fwd_context.get("y_base", torch.zeros_like(y)),
                                  fwd_context.get("x_theta", torch.full_like(x, 2 * math.pi / 3)), fwd_context.get("y_theta", torch.full_like(y, 2 * math.pi / 3)), fwd_context.get("is_av_matmul", False))


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
    'ig_mul': ig_mul,
    'ig_matmul': ig_matmul,
    'slerp_mul': slerp_mul,
    'slerp_matmul': slerp_matmul,
}