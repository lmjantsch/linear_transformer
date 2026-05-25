from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Integration helpers
# ---------------------------------------------------------------------------

_INTEGRATION_STEPS = 5


def fwd_context_hook() -> dict:
    return {}


def _get_slerp(v: torch.Tensor, v_alt: torch.Tensor, alpha: float, theta: torch.Tensor) -> torch.Tensor:
    """Slerp at parameter alpha (0→v, 1→v_alt); falls back to lerp when theta < 1e-5."""
    near_zero = theta.abs() < 1e-5
    safe_sin = torch.sin(theta).abs().clamp(min=1e-8)
    coeff_v   = torch.where(near_zero, torch.full_like(theta, 1.0 - alpha), torch.sin((1.0 - alpha) * theta) / safe_sin)
    coeff_alt = torch.where(near_zero, torch.full_like(theta, alpha),       torch.sin(alpha * theta) / safe_sin)
    return coeff_v * v + coeff_alt * v_alt


def _silu_deriv(x: torch.Tensor) -> torch.Tensor:
    """d/dx [x·σ(x)] = σ(x)·(1 + x·(1 − σ(x)))"""
    s = torch.sigmoid(x)
    return s * (1.0 + x * (1.0 - s))


def _gelu_tanh_deriv(x: torch.Tensor) -> torch.Tensor:
    """Derivative of GELU (tanh approximation)."""
    K, C = math.sqrt(2.0 / math.pi), 0.044715
    inner = K * (x + C * x.pow(3))
    t = torch.tanh(inner)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t.pow(2)) * (K * (1.0 + 3.0 * C * x.pow(2)))


# ---------------------------------------------------------------------------
# Rule 2 variants — zero-preserving nonlinearities
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


# ---------------------------------------------------------------------------
# Lerp / Slerp SiLU  (5-step IG)
# ---------------------------------------------------------------------------

class IntegratedLerpSiLU(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        x_base: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(x, x_base)
        ctx._orig_dtype = x.dtype
        return F.silu(x.float()).to(x.dtype)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_t: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        x, x_base = ctx.saved_tensors
        x_f, base_f = x.float(), x_base.float()
        k = _INTEGRATION_STEPS
        acc = torch.zeros_like(x_f)
        for i in range(1, k + 1):
            acc += _silu_deriv(base_f + (i / k) * (x_f - base_f))
        return (grad_t.float() * acc / k).to(ctx._orig_dtype), None


class IntegratedSlerpSiLU(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        x_base: torch.Tensor,
        theta: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(x, x_base, theta)
        ctx._orig_dtype = x.dtype
        return F.silu(x.float()).to(x.dtype)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_t: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None]:
        x, x_base, theta = ctx.saved_tensors
        x_f, base_f, th_f = x.float(), x_base.float(), theta.float()
        k = _INTEGRATION_STEPS
        acc = torch.zeros_like(x_f)
        for i in range(1, k + 1):
            acc += _silu_deriv(_get_slerp(base_f, x_f, i / k, th_f))
        return (grad_t.float() * acc / k).to(ctx._orig_dtype), None, None


# ---------------------------------------------------------------------------
# Lerp / Slerp GELUTanh  (5-step IG)
# ---------------------------------------------------------------------------

class IntegratedLerpGELUTanh(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        x_base: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(x, x_base)
        ctx._orig_dtype = x.dtype
        return F.gelu(x.float(), approximate="tanh").to(x.dtype)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_t: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        x, x_base = ctx.saved_tensors
        x_f, base_f = x.float(), x_base.float()
        k = _INTEGRATION_STEPS
        acc = torch.zeros_like(x_f)
        for i in range(1, k + 1):
            acc += _gelu_tanh_deriv(base_f + (i / k) * (x_f - base_f))
        return (grad_t.float() * acc / k).to(ctx._orig_dtype), None


class IntegratedSlerpGELUTanh(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        x_base: torch.Tensor,
        theta: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(x, x_base, theta)
        ctx._orig_dtype = x.dtype
        return F.gelu(x.float(), approximate="tanh").to(x.dtype)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_t: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None]:
        x, x_base, theta = ctx.saved_tensors
        x_f, base_f, th_f = x.float(), x_base.float(), theta.float()
        k = _INTEGRATION_STEPS
        acc = torch.zeros_like(x_f)
        for i in range(1, k + 1):
            acc += _gelu_tanh_deriv(_get_slerp(base_f, x_f, i / k, th_f))
        return (grad_t.float() * acc / k).to(ctx._orig_dtype), None, None


# ---------------------------------------------------------------------------
# Rule 3 — Non-zero baseline nonlinearity (Softmax DTD)
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


# ---------------------------------------------------------------------------
# Lerp / Slerp Softmax  (5-step IG)
# ---------------------------------------------------------------------------

class IntegratedLerpSoftmax(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        x_base: torch.Tensor,
        dim: int = -1,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        ctx.save_for_backward(x, x_base)
        ctx._orig_dtype = x.dtype
        ctx._dim = dim
        ctx._dtype = dtype
        return torch.softmax(x, dim=dim, dtype=dtype).to(x.dtype)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_t: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None]:
        x, x_base = ctx.saved_tensors
        dtype, dim = ctx._dtype, ctx._dim
        x_f, base_f = x.to(dtype), x_base.to(dtype)
        k = _INTEGRATION_STEPS
        acc = torch.zeros_like(x_f)
        g = grad_t.to(dtype)
        for i in range(1, k + 1):
            s = torch.softmax(base_f + (i / k) * (x_f - base_f), dim=dim, dtype=dtype)
            acc += s * (g - (g * s).sum(dim=dim, keepdim=True))
        return (acc / k).to(ctx._orig_dtype), None, None, None


class IntegratedSlerpSoftmax(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        x_base: torch.Tensor,
        theta: torch.Tensor,
        dim: int = -1,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        ctx.save_for_backward(x, x_base, theta)
        ctx._orig_dtype = x.dtype
        ctx._dim = dim
        ctx._dtype = dtype
        return torch.softmax(x, dim=dim, dtype=dtype).to(x.dtype)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_t: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None]:
        x, x_base, theta = ctx.saved_tensors
        dtype, dim = ctx._dtype, ctx._dim
        x_f, base_f, th_f = x.to(dtype), x_base.to(dtype), theta.float()
        k = _INTEGRATION_STEPS
        acc = torch.zeros_like(x_f)
        g = grad_t.to(dtype)
        for i in range(1, k + 1):
            x_alpha = _get_slerp(base_f, x_f, i / k, th_f).to(dtype)
            s = torch.softmax(x_alpha, dim=dim, dtype=dtype)
            acc += s * (g - (g * s).sum(dim=dim, keepdim=True))
        return (acc / k).to(ctx._orig_dtype), None, None, None, None


# ---------------------------------------------------------------------------
# Functional wrappers
# ---------------------------------------------------------------------------


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    fwd_context = fwd_context_hook()
    return F.gelu(x, approximate='tanh')


def silu(x: torch.Tensor) -> torch.Tensor:
    fwd_context = fwd_context_hook()
    return F.silu(x)


def softmax(x: torch.Tensor, dim: int = -1, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    fwd_context = fwd_context_hook()
    return F.softmax(x, dim=dim, dtype=dtype)


def secant_gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    fwd_context = fwd_context_hook()
    return SecantGELUTanh.apply(x)


def secant_silu(x: torch.Tensor) -> torch.Tensor:
    fwd_context = fwd_context_hook()
    return SecantSiLU.apply(x)


def integrated_softmax(x: torch.Tensor, dim: int = -1, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    fwd_context = fwd_context_hook()
    return IntegratedSoftmax.apply(x, dim, dtype)


def sec_jac_softmax(x: torch.Tensor, dim: int = -1, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    fwd_context = fwd_context_hook()
    return SecantJacobianSoftmax.apply(x, dim, dtype)


def lerp_silu(x: torch.Tensor) -> torch.Tensor:
    ctx = fwd_context_hook()
    return IntegratedLerpSiLU.apply(x, ctx.get("x_base", torch.zeros_like(x)))


def slerp_silu(x: torch.Tensor) -> torch.Tensor:
    ctx = fwd_context_hook()
    return IntegratedSlerpSiLU.apply(
        x,
        ctx.get("x_base", torch.zeros_like(x)),
        ctx.get("theta", torch.full_like(x, 2 * math.pi / 3)),
    )


def lerp_gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    ctx = fwd_context_hook()
    return IntegratedLerpGELUTanh.apply(x, ctx.get("x_base", torch.zeros_like(x)))


def slerp_gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    ctx = fwd_context_hook()
    return IntegratedSlerpGELUTanh.apply(
        x,
        ctx.get("x_base", torch.zeros_like(x)),
        ctx.get("theta", torch.full_like(x, 2 * math.pi / 3)),
    )


def lerp_softmax(x: torch.Tensor, dim: int = -1, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    ctx = fwd_context_hook()
    return IntegratedLerpSoftmax.apply(x, ctx.get("x_base", torch.zeros_like(x)), dim, dtype)


def slerp_softmax(x: torch.Tensor, dim: int = -1, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    ctx = fwd_context_hook()
    return IntegratedSlerpSoftmax.apply(
        x,
        ctx.get("x_base", torch.zeros_like(x)),
        ctx.get("theta", torch.full_like(x, 2 * math.pi / 3)),
        dim,
        dtype,
    )


# ---------------------------------------------------------------------------
# Mapping wrappers
# ---------------------------------------------------------------------------

ACT_FN = {
    # Standard activations
    'gelu_tanh': gelu_tanh,
    'silu':      silu,
    'softmax':   softmax,
    # Rule 2 — zero-preserving (LVP secant)
    'secant_gelu_tanh': secant_gelu_tanh,
    'secant_silu':      secant_silu,
    # Rule 3 — softmax variants (LVP)
    'sec_jac_softmax':    sec_jac_softmax,
    'integrated_softmax': integrated_softmax,
    # Lerp IG (linear path, external baseline via context hook)
    'lerp_silu':       lerp_silu,
    'lerp_gelu_tanh':  lerp_gelu_tanh,
    'lerp_softmax':    lerp_softmax,
    # Slerp IG (spherical path, external baseline + theta via context hook)
    'slerp_silu':      slerp_silu,
    'slerp_gelu_tanh': slerp_gelu_tanh,
    'slerp_softmax':   slerp_softmax,
}
