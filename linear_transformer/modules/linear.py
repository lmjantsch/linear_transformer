import torch


class LinearAdd(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,  # (..., d)
        other: torch.Tensor,  # (..., d) or scalar
        eps: float
    ) -> torch.Tensor:  # (..., d)
        orig_dtype = x.dtype
        x_f = x.float()
        other_f = other.float()
        
        add_out = x_f + other_f
        
        safe_x = torch.where(x_f >= 0, x_f.clamp(min=eps), x_f.clamp(max=-eps))
        add_ratio = add_out / safe_x

        ctx.save_for_backward(add_ratio.to(orig_dtype))
        ctx._orig_dtype = orig_dtype

        return add_out.to(orig_dtype)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad: torch.Tensor,  # (..., d)
    ) -> tuple[torch.Tensor, torch.Tensor | None, None]:
        (add_ratio,) = ctx.saved_tensors
        add_ratio_f = add_ratio.float()
        grad_f = grad.float()

        grad_x = grad_f * add_ratio_f

        return grad_x.to(ctx._orig_dtype), None, None


class LinearSub(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,  # (..., d)
        other: torch.Tensor,  # (..., d) or scalar
        eps: float
    ) -> torch.Tensor:  # (..., d)
        orig_dtype = x.dtype
        x_f = x.float()
        other_f = other.float()
        
        add_out = x_f - other_f
        
        safe_x = torch.where(x_f >= 0, x_f.clamp(min=eps), x_f.clamp(max=-eps))
        sub_ratio = add_out / safe_x

        ctx.save_for_backward(sub_ratio.to(orig_dtype))
        ctx._orig_dtype = orig_dtype

        return add_out.to(orig_dtype)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad: torch.Tensor,  # (..., d)
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        (sub_ratio,) = ctx.saved_tensors
        sub_ratio_f = sub_ratio.float()
        grad_f = grad.float()
        
        grad_x = grad_f * sub_ratio_f

        return grad_x.to(ctx._orig_dtype), None, None


# ---------------------------------------------------------------------------
# Functional wrappers
# ---------------------------------------------------------------------------

def add(x: torch.Tensor, other: torch.Tensor, eps: float = None) -> torch.Tensor:
    return x + other

def sub(x: torch.Tensor, other: torch.Tensor, eps: float = None) -> torch.Tensor:
    return x - other

def linear_add(x: torch.Tensor, other: torch.Tensor, eps: float) -> torch.Tensor:
    return LinearAdd.apply(x, other, eps)

def linear_sub(x: torch.Tensor, other: torch.Tensor, eps: float) -> torch.Tensor:
    return LinearSub.apply(x, other, eps)


# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------

LINEAR_FN = {
    'add': add,
    'sub': sub,
    'linear_add': linear_add,
    'linear_sub': linear_sub,
}