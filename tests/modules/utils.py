from __future__ import annotations

import pytest
import torch


def get_conservation_error(
    out: torch.Tensor,
    inputs: list[torch.Tensor],
    t: torch.Tensor,
) -> float:
    """|(Σ_i x_i·∇x_i) - (out·t)| / |(out·t)|"""
    lhs = (out.float() * t.float()).sum().item()
    rhs = sum(
        (x.float() * x.grad.float()).sum().item()
        for x in inputs
        if x.grad is not None
    )
    return abs(lhs - rhs) / (abs(lhs) + 1e-10)


def register_error(module_name: str, error: float) -> None:
    """Register a conservation error for the summary table."""
    if not hasattr(pytest, "_module_error_results"):
        pytest._module_error_results = {}
    pytest._module_error_results[module_name] = error