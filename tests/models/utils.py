from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from nnsight import NNsight

from modular_transformer import patch_model_for_lvp


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
    return _LLAMA_LIKE  # LLaMA, Qwen, Mistral, etc


# -------------------------------------------------------
# Metrics
# -------------------------------------------------------

@torch.no_grad()
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


@torch.no_grad()
def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative L2 error: ||a - b|| / (||a|| + 1e-10)."""
    return ((a.float() - b.float()).norm() / (a.float().norm() + 1e-10)).item()

@torch.no_grad()
def get_residual_errors(model_cache: dict, baseline_cache: dict, num_layers: int):
    errors: dict[str, list[float]] = {"mid": [], "out": []}

    for i in range(num_layers):
        errors["mid"].append(
            rel_l2(baseline_cache[i]["mid"], model_cache[i]["mid"])
        )
        errors["out"].append(
            rel_l2(baseline_cache[i]["out"], model_cache[i]["out"])
        )
    return errors


@torch.no_grad()
def get_residual_gradient_errors(model_cache: dict, baseline_cache: dict, num_layers: int):
    errors: dict[str, list[float]] = {"mid_grad": [], "out_grad": []}

    for i in range(num_layers - 1, -1, -1):
        errors["mid_grad"].append(
            rel_l2(baseline_cache[i]["mid_grad"], model_cache[i]["mid_grad"])
        )
        errors["out_grad"].append(
            rel_l2(baseline_cache[i]["out_grad"], model_cache[i]["out_grad"])
        )
    return errors

@torch.no_grad()
def get_kl_values_from_logits(model_cache, baseline_cache):
    p_logits = baseline_cache['logits'] # Shape: [B, S, V]
    q_logits = model_cache['logits']    # Shape: [B, S, V]

    p = F.softmax(p_logits, dim=-1) + 1e-7
    q = F.softmax(q_logits, dim=-1) + 1e-7

    seq_mean_kl_all = (p * (p.log() - q.log())).sum(dim=-1).mean(dim=-1)
    kl_values = seq_mean_kl_all.tolist()

    return kl_values




# -------------------------------------------------------
# Caching 
# -------------------------------------------------------

# model = AutoModelForCausalLM.from_pretrained(
#         model_id, torch_dtype=dtype, attn_implementation='eager'
#     ).to(device).eval()
#     model = patch_model_for_lvp(model, **lvp_config)

def detach_and_move_to_device(cache: dict, device: str | torch.device):
    if isinstance(device, str):
        device = torch.device(device)

    for key in list(cache):
        if isinstance(cache[key], dict):
            detach_and_move_to_device(cache[key], device)
        elif isinstance(cache[key], torch.Tensor):
            cache[key] = cache[key].detach().to(device)
        else:
            continue


def cache_forward_and_backward(
    model_id: str,
    model: AutoModelForCausalLM,
    batch_inputs: dict,
    cache: dict,
) -> dict:
    arc = get_arch(model_id)
    layers = arc.layers(model)
    
    num_layers = len(layers)
    cache: dict = {i: {} for i in range(num_layers)}

    try:
        with model.trace(**batch_inputs) as tracer:
            cache['emb'] = layers[0].input.save()

            for i, layer in enumerate(layers):
                cache[i]['ln1_in']   = getattr(layer, arc.ln1).input.save()
                cache[i]['ln1_out']  = getattr(layer, arc.ln1).output.save()
                cache[i]['attn_out'] = getattr(layer, arc.attn).output[0].save()
                cache[i]['mid']      = getattr(layer, arc.ln2).input.save()
                cache[i]['ln2_in']   = getattr(layer, arc.ln2).input.save()
                cache[i]['ln2_out']  = getattr(layer, arc.ln2).output.save()
                cache[i]['mlp_out']  = getattr(layer, arc.mlp).output.save()
                cache[i]['out']      = layer.output.save()

            logits = model.lm_head.output
            cache['logits'] = logits.save()
            loss = logits[:, -1].max(dim=-1).values.sum()
            cache['loss'] = loss.save()

            with loss.backward():
                for j in range(num_layers - 1, -1, -1):
                    for k in list(cache[j])[::-1]:
                        if k == 'ln1_in': # subtract skip-connection grad to isolate the attention path
                            cache[j][k + '_grad'] = (
                                cache[j][k].grad.save() - cache[j]['mid_grad']
                            ).detach()
                        elif k == 'ln2_in': # subtract skip-connection grad to isolate the MLP path
                            cache[j][k + '_grad'] = (
                                cache[j][k].grad.save() - cache[j]['out_grad']
                            ).detach()
                        else:
                            cache[j][k + '_grad'] = cache[j][k].grad.save().detach()
                cache['emb_grad'] = cache['emb'].grad.save()
    finally:
        detach_and_move_to_device(cache, 'cpu')
    return cache


# -------------------------------------------------------
# Print result tables
# -------------------------------------------------------


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

def print_forward_identity_table(results: dict[str, dict]) -> None:
    """Cross-model summary: max forward residual divergence for identity-patched models."""
    if not results:
        return
    model_w = 32
    col_w = 12
    header = f"{'Model':<{model_w}s}{'max(mid)':>{col_w}s}{'max(out)':>{col_w}s}  Status"
    sep = "-" * len(header)
    print("\n" + "=" * len(header))
    print("FORWARD IDENTITY (rel L2): identity-patched vs original")
    print("=" * len(header))
    print(header)
    print(sep)
    for mid in sorted(results.keys()):
        data = results[mid]
        max_mid = max(data["errors"]["mid"])
        max_out = max(data["errors"]["out"])
        status = "✓ PASS" if data["passed"] else "✗ FAIL"
        print(f"{mid:<{model_w}s}{max_mid:>{col_w}.2e}{max_out:>{col_w}.2e}  {status}")
    print("=" * len(header) + "\n")


def print_backward_identity_table(results: dict[str, dict]) -> None:
    """Cross-model summary: max gradient divergence for identity-patched models."""
    if not results:
        return
    model_w = 32
    col_w = 14
    header = f"{'Model':<{model_w}s}{'max(mid_grad)':>{col_w}s}{'max(out_grad)':>{col_w}s}  Status"
    sep = "-" * len(header)
    print("\n" + "=" * len(header))
    print("BACKWARD IDENTITY (rel L2): identity-patched vs original")
    print("=" * len(header))
    print(header)
    print(sep)
    for mid in sorted(results.keys()):
        data = results[mid]
        max_mid_grad = max(data["errors"]["mid_grad"])
        max_out_grad = max(data["errors"]["out_grad"])
        status = "✓ PASS" if data["passed"] else "✗ FAIL"
        print(f"{mid:<{model_w}s}{max_mid_grad:>{col_w}.2e}{max_out_grad:>{col_w}.2e}  {status}")
    print("=" * len(header) + "\n")


def print_emb_grad_identity_table(results: dict[str, float]) -> None:
    """Cross-model summary: embedding gradient identity error."""
    if not results:
        return
    model_w = 32
    col_w = 14
    header = f"{'Model':<{model_w}s}{'emb_grad err':>{col_w}s}  Status"
    sep = "-" * len(header)
    print("\n" + "=" * len(header))
    print("EMBEDDING GRADIENT IDENTITY (rel L2): identity-patched vs original")
    print("=" * len(header))
    print(header)
    print(sep)
    for mid in sorted(results.keys()):
        err = results[mid]
        status = "✓ PASS" if err < 1e-2 else "✗ FAIL"
        print(f"{mid:<{model_w}s}{err:>{col_w}.2e}  {status}")
    print("=" * len(header) + "\n")


def print_conservation_summary_table(results: dict[str, dict]) -> None:
    """Cross-config summary: max per-module conservation error for LVP-patched models."""
    if not results:
        return
    modules = ["ln1", "attn", "ln2", "mlp"]
    model_w = 35
    col_w = 10
    header = f"{'Config':<{model_w}s}" + "".join(f"{m:>{col_w}s}" for m in modules) + "  Status"
    sep = "-" * len(header)
    print("\n" + "=" * len(header))
    print("MODULE CONSERVATION MAX ERROR (per config)")
    print("=" * len(header))
    print(header)
    print(sep)
    for mid in sorted(results.keys()):
        data = results[mid]
        status = "✓ PASS" if data["passed"] else "✗ FAIL"
        row = f"{mid:<{model_w}s}"
        for m in modules:
            row += f"{max(data['errors'][m]):>{col_w}.2e}"
        row += f"  {status}"
        print(row)
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