from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _secant_denom(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Stabilised denominator for Rule 2: x + eps * sign(x), clamped so sign never flips."""
    return x + eps * x.sign().clamp(min=1)

# ---------------------------------------------------------------------------
# MLP activations
# ---------------------------------------------------------------------------

class SecantGELUTanh(torch.autograd.Function):
    """Rule 2 — GELU (tanh approximation, used in Gemma2).

    Secant: c = 0.5 * (1 + tanh(√(2/π) * (x + 0.044715·x³)))
    """

    _SQRT_2_OVER_PI: float = math.sqrt(2.0 / math.pi)

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,  # (..., d)
    ) -> torch.Tensor:  # (..., d)
        orig_dtype = x.dtype
        ctx.save_for_backward(x)
        ctx._orig_dtype = orig_dtype
        return F.gelu(x.float(), approximate="tanh").to(orig_dtype)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_t: torch.Tensor,  # (..., d)
    ) -> torch.Tensor:  # (..., d)
        (x,) = ctx.saved_tensors
        x_f = x.float()
        inner = SecantGELUTanh._SQRT_2_OVER_PI * (x_f + 0.044715 * x_f.pow(3))
        c = 0.5 * (1.0 + torch.tanh(inner))
        return (grad_t.float() * c).to(ctx._orig_dtype)


class SecantSiLU(torch.autograd.Function):
    """Rule 2 — SiLU / Swish.

    Secant: c = sigmoid(x)
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,  # (..., d)
    ) -> torch.Tensor:  # (..., d)
        orig_dtype = x.dtype
        ctx.save_for_backward(x)
        ctx._orig_dtype = orig_dtype
        return F.silu(x.float()).to(orig_dtype)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_t: torch.Tensor,  # (..., d)
    ) -> torch.Tensor:  # (..., d)
        (x,) = ctx.saved_tensors
        c = torch.sigmoid(x.float())
        return (grad_t.float() * c).to(ctx._orig_dtype)

def secant_gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    return SecantGELUTanh.apply(x)


def secant_silu(x: torch.Tensor) -> torch.Tensor:
    return SecantSiLU.apply(x)

# ---------------------------------------------------------------------------
# softmax
# ---------------------------------------------------------------------------

class IntegratedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,  # (..., N)
        dim: int = -1,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        s = torch.softmax(x, dim=dim, dtype=dtype)
        
        ctx.save_for_backward(x)
        ctx._orig_dtype = orig_dtype
        ctx._dim = dim
        ctx._dtype = dtype
        
        return s.to(orig_dtype)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_t: torch.Tensor,  # (..., N)
    ) -> tuple[torch.Tensor, None, None]:
        (x,) = ctx.saved_tensors
        dtype = ctx._dtype
        dim = ctx._dim

        k_steps = 10
        
        grad_t_f = grad_t.to(dtype=dtype)
        accumulated_grads = torch.zeros_like(x, dtype=dtype)
        
        # Generate alphas: [1/k, 2/k, ..., 1.0]
        alphas = torch.linspace(1.0 / k_steps, 1.0, steps=k_steps, device=x.device, dtype=dtype)
        baseline = torch.zeros_like(x, dtype=x.dtype, device=x.device)
        for alpha in alphas:
            x_alpha = baseline + alpha * (x - baseline)
            s_alpha = torch.softmax(x_alpha, dim=dim, dtype=dtype)
            
            dot_product = (grad_t_f * s_alpha).sum(dim=dim, keepdim=True)
            step_grad = s_alpha * (grad_t_f - dot_product)
            accumulated_grads += step_grad
            
        # Average the accumulated gradients
        result = accumulated_grads / k_steps

        return result.to(ctx._orig_dtype), None, None


class SecantJacobianSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,  # (..., N)
        dim: int = -1,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        s = torch.softmax(x, dim=dim, dtype=dtype)
        # We only need the raw inputs for the analytical backward pass!
        ctx.save_for_backward(x.to(dtype)) 
        ctx._orig_dtype = orig_dtype
        ctx._dim = dim
        ctx._dtype = dtype
        return s.to(orig_dtype)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_t: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None]:
        (x_f,) = ctx.saved_tensors
        dim = ctx._dim
        dtype = ctx._dtype
        
        grad_f = grad_t.to(dtype=dtype)
        N = x_f.shape[dim]
        
        # If sequence length / vocab size is 1, the gradient is always 0
        if N <= 1:
            return torch.zeros_like(grad_t), None, None

        # --- Analytical Secant Factor Calculation ---
        # f(x) = (exp(x) - 1) / (x * (N - 1 + exp(x)))
        
        # 1. Taylor Expansion for |x| near 0 (Prevents 0/0 NaN)
        # f(x) ≈ 1/N + (N-2)/(2N^2) * x
        eps = 1e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-4
        taylor_approx = 1.0 / N + ((N - 2.0) / (2.0 * N * N)) * x_f
        
        # 2. Stable formula for x > 0 (Prevents exp(x) inf overflow)
        exp_neg_x = torch.exp(-x_f)
        factor_pos = (1.0 - exp_neg_x) / (x_f * ((N - 1) * exp_neg_x + 1.0))
        
        # 3. Stable formula for x < 0 (Prevents underflow issues)
        exp_x = torch.exp(x_f)
        factor_neg = (exp_x - 1.0) / (x_f * (N - 1 + exp_x))
        
        # Combine domains seamlessly
        factor = torch.where(x_f > 0, factor_pos, factor_neg)
        factor = torch.where(torch.abs(x_f) < eps, taylor_approx, factor)
        
        # --- Vector-Jacobian Product ---
        G_sum = grad_f.sum(dim=dim, keepdim=True)
        result = factor * (grad_f - G_sum / N)
        
        return result.to(ctx._orig_dtype), None, None


def integrated_softmax(x: torch.Tensor, dim: int = -1, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return IntegratedSoftmax.apply(x, dim, dtype)

def sec_jac_softmax(x: torch.Tensor, dim: int = -1, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return SecantJacobianSoftmax.apply(x, dim, dtype)

# ---------------------------------------------------------------------------
# Mapping wrappers
# ---------------------------------------------------------------------------

ACT_FN = {
    # Standard activations
    'gelu_tanh':      lambda x: F.gelu(x, approximate='tanh'),
    'silu':           F.silu,
    'softmax':        F.softmax,
    # Secant MLP activations
    'secant_gelu_tanh': secant_gelu_tanh,
    'secant_silu':      secant_silu,
    # Custom Softmax activations
    'sec_jac_softmax':    sec_jac_softmax,
    'integrated_softmax': integrated_softmax,
}