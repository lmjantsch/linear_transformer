from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch


@dataclass
class ArchAccessors:
    """Layer accessors for different transformer architectures."""

    embed: Callable        # (model) -> embed module
    layers: Callable       # (model) -> layer list
    ln1: str               # pre-attention norm attribute
    attn: str              # attention attribute
    ln2: str               # pre-MLP norm attribute
    mlp: str               # MLP attribute


_LLAMA_LIKE = ArchAccessors(
    embed=lambda m: m.model.embed_tokens,
    layers=lambda m: m.model.layers,
    ln1="input_layernorm",
    attn="self_attn",
    ln2="post_attention_layernorm",
    mlp="mlp",
)

_GPT2 = ArchAccessors(
    embed=lambda m: m.transformer.wte,
    layers=lambda m: m.transformer.h,
    ln1="ln_1",
    attn="attn",
    ln2="ln_2",
    mlp="mlp",
)

_GEMMA2 = ArchAccessors(
    embed=lambda m: m.model.embed_tokens,
    layers=lambda m: m.model.layers,
    ln1="input_layernorm",
    attn="self_attn",
    ln2="pre_feedforward_layernorm",
    mlp="mlp",
)


def get_arch(model_id: str) -> ArchAccessors:
    """Map model ID to architecture accessors."""
    mid = model_id.lower()
    if "gpt" in mid:
        return _GPT2
    if "gemma" in mid:
        return _GEMMA2
    return _LLAMA_LIKE  # LLaMA, Qwen, Mistral, etc.

def rel_err(
    x: torch.Tensor,
    gx: torch.Tensor,
    y: torch.Tensor,
    gy: torch.Tensor,
) -> float:
    """Relative conservation error: |(x·gx).sum() - (y·gy).sum()| / |(x·gx).sum()|."""
    s_in  = (x.float() * gx.float()).sum().item()
    s_out = (y.float() * gy.float()).sum().item()
    return abs(s_in - s_out) / abs(s_in + 1e-10)


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative L2 error: ||a - b|| / (||a|| + 1e-10)."""
    return ((a.float() - b.float()).norm() / (a.float().norm() + 1e-10)).item()


def print_forward_table(
    model_id: str,
    errors: dict[str, list[float]],
) -> None:
    """Print per-layer forward divergence (rel L2) table for one model."""
    num_layers = len(next(iter(errors.values())))
    col_w = 10
    points = list(errors.keys())

    header = f"{'Layer':>6s}" + "".join(f"{p:>{col_w}s}" for p in points)
    sep = "-" * len(header)

    print(f"\n{'='*len(header)}")
    print(f"FORWARD DIVERGENCE (rel L2): {model_id}")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)

    for i in range(num_layers):
        row = f"{i:>6d}" + "".join(f"{errors[p][i]:>{col_w}.2e}" for p in points)
        print(row)

    print(sep)
    for stat, fn in [("mean", np.mean), ("std", np.std), ("max", np.max)]:
        row = f"{stat:>6s}" + "".join(f"{fn(errors[p]):>{col_w}.2e}" for p in points)
        print(row)
    print("=" * len(header) + "\n")


def print_component_table(
    model_id: str,
    errors: dict[str, list[float]],
) -> None:
    """Print per-layer and aggregate conservation error table for one model."""
    num_layers = len(next(iter(errors.values())))
    col_w = 10
    modules = ["ln1", "attn", "ln2", "mlp"]

    header = f"{'Layer':>6s}" + "".join(f"{m:>{col_w}s}" for m in modules)
    sep = "-" * len(header)

    print(f"\n{'='*len(header)}")
    print(f"CONSERVATION ANALYSIS: {model_id}")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)

    for i in range(num_layers):
        row = f"{i:>6d}" + "".join(f"{errors[m][i]:>{col_w}.4f}" for m in modules)
        print(row)

    print(sep)
    for stat, fn in [("mean", np.mean), ("std", np.std), ("max", np.max)]:
        row = f"{stat:>6s}" + "".join(f"{fn(errors[m]):>{col_w}.2e}" for m in modules)
        print(row)
    print("=" * len(header) + "\n")


def print_conservation_embedding_table(results: dict[str, float]) -> None:
    """Print per-model embedding conservation ratio summary."""
    if not results:
        return

    model_w = 25
    col_w = 12
    header = f"{'Model':<{model_w}s}{'Ratio':>{col_w}s}{'|ratio-1|':>{col_w}s}  Status"
    sep = "-" * len(header)

    print("\n" + "=" * len(header))
    print("EMBEDDING CONSERVATION: (emb · ∇emb).sum() / loss  (ideal = 1.0)")
    print("=" * len(header))
    print(header)
    print(sep)

    for mid in sorted(results.keys()):
        ratio = results[mid]
        dev = abs(ratio - 1.0)
        status = "✓ PASS" if dev < 0.1 else "✗ FAIL"
        print(f"{mid:<{model_w}s}{ratio:>{col_w}.2e}{dev:>{col_w}.2e}  {status}")

    print("=" * len(header) + "\n")


def print_kl_table(results: dict[str, list[float]]) -> None:
    if not results:
        return

    num_prompts = max(len(v) for v in results.values())
    col_w = 11
    model_w = 25

    header = f"{'Model':<{model_w}s}" + "".join(f"{'P'+str(i):>{col_w}s}" for i in range(num_prompts))
    header += f"{'Max KL':>{col_w}s}  Status"
    sep = "-" * len(header)

    print("\n" + "=" * len(header))
    print("KL DIVERGENCE AFTER PATCHING (per model, per prompt)")
    print("=" * len(header))
    print(header)
    print(sep)

    for model_name in sorted(results.keys()):
        kls = results[model_name]
        max_kl = max(kls) if kls else 0.0
        status = "✓ PASS" if max_kl < 5e-3 else "✗ FAIL"

        row = f"{model_name:<{model_w}s}"
        for kl in kls:
            row += f"{kl:>{col_w}.2e}"
        row += f"{max_kl:>{col_w}.2e}  {status}"
        print(row)

    print("=" * len(header) + "\n")