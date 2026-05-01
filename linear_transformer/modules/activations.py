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
# Rule 2 variants — zero-preserving nonlinearities
# ---------------------------------------------------------------------------

class SecantGELU(torch.autograd.Function):
    """Rule 2 — GELU (erf formulation).

    Secant: c = Φ(x) = 0.5 * (1 + erf(x / √2))
    Forward:  f(x_i) = x_i * c
    Backward: f⁻¹(t) = t * c
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,  # (..., d)
    ) -> torch.Tensor:  # (..., d)
        orig_dtype = x.dtype
        ctx.save_for_backward(x)
        ctx._orig_dtype = orig_dtype
        return F.gelu(x.float()).to(orig_dtype)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_t: torch.Tensor,  # (..., d)
    ) -> torch.Tensor:  # (..., d)
        (x,) = ctx.saved_tensors
        c = 0.5 * (1.0 + torch.erf(x.float() / math.sqrt(2.0)))  # Gaussian CDF
        return (grad_t.float() * c).to(ctx._orig_dtype)


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


class SecantReLU(torch.autograd.Function):
    """Rule 2 — ReLU.

    Secant: c = relu(x) / (x + eps·sign(x).clamp(min=1))
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,  # (..., d)
        eps: float = 1e-6,
    ) -> torch.Tensor:  # (..., d)
        orig_dtype = x.dtype
        ctx.save_for_backward(x)
        ctx._orig_dtype = orig_dtype
        ctx._eps = eps
        return F.relu(x.float()).to(orig_dtype)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_t: torch.Tensor,  # (..., d)
    ) -> tuple[torch.Tensor, None]:
        (x,) = ctx.saved_tensors
        x_f = x.float()
        c = F.relu(x_f) / _secant_denom(x_f, ctx._eps)
        return (grad_t.float() * c).to(ctx._orig_dtype), None

class SecantTanh(torch.autograd.Function):
    
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,  # (..., d)
    ) -> torch.Tensor:  # (..., d)
        orig_dtype = x.dtype
        ctx.save_for_backward(x)
        ctx._orig_dtype = orig_dtype
        return torch.tanh(x.float()).to(orig_dtype)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_t: torch.Tensor,  # (..., d)
    ) -> torch.Tensor:  # (..., d)
        (x,) = ctx.saved_tensors
        x_f = x.float()
        c = torch.ones_like(x_f)
        mask = torch.abs(x_f) > 1e-6
        c[mask] = torch.tanh(x_f[mask]) / x_f[mask]
        return (grad_t.float() * c).to(ctx._orig_dtype)


# ---------------------------------------------------------------------------
# Rule 3 — Non-zero baseline nonlinearity (Softmax DTD)
# ---------------------------------------------------------------------------

class DTDSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,  # (..., N)
        dim: int = -1,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:  # (..., N)
        orig_dtype = x.dtype
        s = torch.softmax(x, dim=dim, dtype=dtype)
        ctx.save_for_backward(s.to(orig_dtype))
        ctx._orig_dtype = orig_dtype
        ctx._dim = dim
        ctx._dtype = dtype
        return s.to(orig_dtype)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_t: torch.Tensor,  # (..., N)
    ) -> tuple[torch.Tensor, None, None]:
        (s,) = ctx.saved_tensors
        dim = ctx._dim
        dtype = ctx._dtype
        s_f = s.to(dtype=dtype)
        # f⁻¹(t) = t - s · (Σ_j t_j)
        result = grad_t.to(dtype=dtype) - s_f * grad_t.to(dtype=dtype).sum(dim=dim, keepdim=True)
        return result.to(ctx._orig_dtype), None, None  # no gradient w.r.t. dim

class SecantSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,  # (..., N)
        dim: int = -1,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:  # (..., N)
        orig_dtype = x.dtype
        s = torch.softmax(x, dim=dim, dtype=dtype)
        ctx.save_for_backward(s.to(orig_dtype), x)
        ctx._orig_dtype = orig_dtype
        ctx._dtype = dtype
        return s.to(orig_dtype)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_t: torch.Tensor,  # (..., N)
    ) -> tuple[torch.Tensor, None, None]:
        (s, x) = ctx.saved_tensors
        dtype = ctx._dtype
        s_f = s.to(dtype=dtype)
        x_f = x.to(dtype=dtype)
        result = s_f / (x_f + 1e-07) * grad_t.to(dtype=dtype).sum(-1, keepdim=True)
        return result.to(ctx._orig_dtype), None, None 
    
class PosRationSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,  # (..., N)
        dim: int = -1,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:  # (..., N)
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
        (x, ) = ctx.saved_tensors
        dtype = ctx._dtype
        x_f = x.to(dtype=dtype).clip(min=0.0)
        result = x_f / (x_f.sum(dim=-1, keepdim=True) + 1e-7) * grad_t.to(dtype=dtype)
        return result.to(ctx._orig_dtype), None, None  # no gradient w.r.t. dim

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

import torch

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


class FrozenDenomSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,  # (..., N)
        dim: int = -1,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f = x.float()
        x_max = x_f.max(dim=dim, keepdim=True).values
        exps = torch.exp(x_f - x_max)
        sum_exps = torch.sum(exps, dim=dim, keepdim=True)
    
        s = exps / sum_exps
        
        ctx.save_for_backward(sum_exps)
        ctx._orig_dtype = orig_dtype
        ctx._dtype = dtype
        
        return s.to(orig_dtype)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_t: torch.Tensor,  # (..., N)
    ) -> tuple[torch.Tensor, None, None]:
        (sum_exps,) = ctx.saved_tensors
        dtype = ctx._dtype
        
        grad_t_f = grad_t.to(dtype)

        result = torch.log(grad_t_f.clip(min=1e-7) * sum_exps)

        return result.to(ctx._orig_dtype), None, None

class OuterProdSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,  # (..., N)
        dim: int = -1,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f = x.to(dtype) # Safer to use .to() instead of .float() if dtype arg is passed
        
        # Standard stable Softmax
        x_max = x_f.max(dim=dim, keepdim=True).values
        exps = torch.exp(x_f - x_max)
        sum_exps = torch.sum(exps, dim=dim, keepdim=True)
        s = exps / sum_exps
        
        # Save tensors and attributes needed for backward
        ctx.save_for_backward(x, s.to(orig_dtype))
        ctx._orig_dtype = orig_dtype
        ctx._dtype = dtype
        ctx._dim = dim  # Save the dimension!
        
        return s.to(orig_dtype)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_t: torch.Tensor,  # The incoming vector t from the next layer
    ) -> tuple[torch.Tensor, None, None, None]: 
        # Note: Return a None for every input argument in forward (x, dim, dtype)
        
        (x, s) = ctx.saved_tensors
        dim = ctx._dim
        dtype = ctx._dtype
        
        x_f = x.to(dtype)
        s_f = s.to(dtype)
        grad_t_f = grad_t.to(dtype)

        s_dot_t = torch.sum(s_f * grad_t_f, dim=dim, keepdim=True)
        
        norm_sq = torch.sum(x_f * x_f, dim=dim, keepdim=True)
        norm_sq = torch.clamp(norm_sq, min=1e-12)
        
        result = (s_dot_t / norm_sq) * x_f

        return result.to(ctx._orig_dtype), None, None

# ---------------------------------------------------------------------------
# Functional wrappers
# ---------------------------------------------------------------------------

def secant_gelu(x: torch.Tensor) -> torch.Tensor:
    return SecantGELU.apply(x)


def secant_gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    return SecantGELUTanh.apply(x)


def secant_silu(x: torch.Tensor) -> torch.Tensor:
    return SecantSiLU.apply(x)


def secant_relu(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return SecantReLU.apply(x, eps)


def secant_tanh(x: torch.Tensor) -> torch.Tensor:
    return SecantTanh.apply(x)


def dtd_softmax(x: torch.Tensor, dim: int = -1, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return DTDSoftmax.apply(x, dim, dtype)


def secant_softmax(x: torch.Tensor, dim: int = -1, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return SecantSoftmax.apply(x, dim, dtype)


def pos_ratio_softmax(x: torch.Tensor, dim: int = -1, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return PosRationSoftmax.apply(x, dim, dtype)


def integrated_softmax(x: torch.Tensor, dim: int = -1, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return IntegratedSoftmax.apply(x, dim, dtype)


def sec_jac_softmax(x: torch.Tensor, dim: int = -1, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return SecantJacobianSoftmax.apply(x, dim, dtype)

def frozen_denom_softmax(x: torch.Tensor, dim: int = -1, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return FrozenDenomSoftmax.apply(x, dim, dtype)

def outer_prod_softmax(x: torch.Tensor, dim: int = -1, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return OuterProdSoftmax.apply(x, dim, dtype)

def constant_softmax(x: torch.Tensor, dim: int = -1, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return F.softmax(x, dim=dim, dtype=dtype).detach()


# ---------------------------------------------------------------------------
# Mapping wrappers
# ---------------------------------------------------------------------------

ACT_FN = {
    # Standard activations
    'gelu':           F.gelu,
    'gelu_tanh':      lambda x: F.gelu(x, approximate='tanh'),
    'silu':           F.silu,
    'relu':           F.relu,
    'tanh':           torch.tanh,
    'softmax':        F.softmax,
    # Rule 2 — zero-preserving (LVP secant)
    'secant_gelu':      secant_gelu,
    'secant_gelu_tanh': secant_gelu_tanh,
    'secant_silu':      secant_silu,
    'secant_relu':      secant_relu,
    'secant_tanh':      secant_tanh,
    # Rule 3 — softmax variants (LVP)
    'dtd_softmax':        dtd_softmax,
    'sec_jac_softmax':    sec_jac_softmax,
    'integrated_softmax': integrated_softmax,
    'secant_softmax':     secant_softmax,
    'pos_ratio_softmax':  pos_ratio_softmax,
    'frozen_denom_softmax': frozen_denom_softmax,
    'outer_prod_softmax': outer_prod_softmax,
    'constant_softmax':   constant_softmax,
}