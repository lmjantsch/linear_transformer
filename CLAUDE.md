# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research implementation of **Layer-Wise Vector Propagation (LVP)**, a method for attributing transformer model outputs to decomposed input components. The goal is to linearize a neural network so that input components $\mathbf{x}_i$ (where $\mathbf{x} = \sum_i \mathbf{x}_i$) can be propagated independently, satisfying the symmetry condition:

$$f(\mathbf{x}_i) \bullet \mathbf{t} = \mathbf{x}_i \bullet f^{-1}(\mathbf{t})$$

**Research documentation** lives in `research_paper_md/math.md`.

**Implementation** lives in `linear_transformer/`.

## Core Mathematical Rules

| Rule | Layer Type | Key Idea |
|------|-----------|----------|
| Rule 1 | Linear (matmul) | Bias treated as own component; backward uses $\mathbf{W}^T$ |
| Rule 2 | Zero-preserving nonlinearities (ReLU, GELU, SiLU, tanh) | Element-wise secant; simplifies to $\Phi(\mathbf{x})$ (GELU), $\sigma(\mathbf{x})$ (SiLU), $\tanh(x)/x$ (tanh) |
| Rule 3 | Non-zero baseline nonlinearities (Softmax) | Deep Taylor Decomposition; Jacobian $\mathbf{I} - \mathbf{1}\mathbf{s}^T$ |
| Rule 4 | Bilinear (element-wise product or matmul) | Uniform splitting: each operand gets $\frac{1}{2}$ the relevance |

Attention applies Rule 4 twice (at $\mathbf{Q}\mathbf{K}^T$ and $\mathbf{A}\mathbf{V}$), yielding natural splits: $\mu_v = \frac{1}{2}$, $\mu_q = \frac{1}{4}$, $\mu_k = \frac{1}{4}$.

## How LVP Works

Rules 2–4 are implemented as custom `torch.autograd.Function` nodes. Their `forward` executes standard math; their `backward` propagates $\mathbf{t}$ via the LVP rule instead of the true gradient.

Rule 1 (linear layers) and norm layers are handled structurally: `Frozen*` wrappers detach the normalization factor and call the underlying `nn.Linear` directly — standard autograd then produces $\mathbf{W}^T \mathbf{t}$ automatically.

**Usage:**
```python
from linear_transformer import patch_model_for_lvp

model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation='eager').eval()
model = patch_model_for_lvp(model)  # returns NNsight-wrapped model

inputs = tokenizer("Paris is the capital of", return_tensors="pt")
with model.trace(**inputs):
    emb = model.model.embed_tokens.output.save()
    logits = model.lm_head.output[:, -1].save()
    loss = logits.max(dim=-1).values.sum()
    with loss.backward():
        emb_grad = emb.grad.save()

# Component attribution: (emb · emb_grad).sum() ≈ loss
```

## Code Structure

```
linear_transformer/
├── __init__.py            # exports patch_model_for_lvp, register_lvp_module
├── modules/
│   ├── activations.py     # Rule 2/3 autograd.Function nodes + functional wrappers
│   └── bilinear.py        # BilinearMul, BilinearMatmul (Rule 4) + functional wrappers
├── models/
│   ├── generic.py         # FrozenLayerNorm
│   ├── gpt2.py            # FrozenGPT2MLP, FrozenGPT2Attention
│   ├── llama2.py          # FrozenLlama2RMSNorm, FrozenLlama2SwiGLU, FrozenLlama2Attention
│   └── gemma2.py          # FrozenGemma2RMSNorm, FrozenGemma2GeGLU, FrozenGemma2Attention
└── patching/
    ├── registry.py        # _REGISTRY; register_lvp_module(); get_registry()
    └── patcher.py         # patch_model_for_lvp(model, include, exclude, nnsight_wrapper, dry_run)
```

## Supported Models

| Model | Attn wrapper | MLP wrapper | Norm wrapper |
|-------|-------------|------------|-------------|
| GPT-2 | `FrozenGPT2Attention` (no RoPE, fused c_attn) | `FrozenGPT2MLP` (tanh-GELU) | `FrozenLayerNorm` |
| LLaMA 3.1 | `FrozenLlama2Attention` (RoPE + GQA) | `FrozenLlama2SwiGLU` (SiLU gate) | `FrozenLlama2RMSNorm` |
| Qwen2.5 | `FrozenLlama2Attention` (RoPE + GQA) | `FrozenLlama2SwiGLU` (SiLU gate) | `FrozenLlama2RMSNorm` |
| Gemma2 | `FrozenGemma2Attention` (RoPE + GQA + softcap) | `FrozenGemma2GeGLU` (tanh-GELU gate) | `FrozenGemma2RMSNorm` |

## Key Implementation Details

- **Weight sharing:** `from_module(m)` stores references to original parameters — no weight copies, zero memory overhead.
- **Precision:** All rule computations cast to float32 internally, cast back to original dtype on return.
- **Norm linearization:** `FrozenRMSNorm`/`FrozenLayerNorm` compute the normalization factor under `float32` and detach it, making the layer linear in `x` so standard autograd gives $\mathbf{W}^T \mathbf{t}$.
- **Attention softmax:** Configurable via the `attn_act_fn` kwarg to `patch_model_for_lvp`. Default is `dtd_softmax`. When `attn_weights.grad_fn` is `None` (no-grad context), the AV product falls back to plain `torch.matmul`.
- **GQA:** `repeat_kv` expands K/V heads before attention (LLaMA/Gemma2).
- **Gemma2 softcapping:** logit softcap uses `secant_tanh` (Rule 2).
- **NNsight integration:** `patch_model_for_lvp` wraps the patched model in `NNsight` by default (`nnsight_wrapper=True`). Disable with `nnsight_wrapper=False`.
- **Extending to new models:** Call `register_lvp_module(MyHFClass, MyFrozenWrapper.from_module)` before `patch_model_for_lvp`.

## Attention Softmax Variants

All live in `linear_transformer/modules/activations.py` and are importable from `linear_transformer.modules.activations`:

| Function | Rule | Backward |
|----------|------|----------|
| `dtd_softmax` | Rule 3 DTD (default) | $\mathbf{t} - \mathbf{s}(\mathbf{s}^T\mathbf{t})$ — exactly conservation-preserving |
| `sec_jac_softmax` | Secant-Jacobian (analytic) | analytic closed-form; stable across all x |
| `integrated_softmax` | Integrated gradients (10 steps) | average Jacobian along path from 0 |
| `secant_softmax` | Element-wise secant | $\mathbf{s}/(x + \epsilon) \cdot \sum t_j$ |
| `pos_ratio_softmax` | Positive-ratio | clips x to positive, uniform split |
| `constant_softmax` | Constant (detached) | treats attention weights as a constant |
