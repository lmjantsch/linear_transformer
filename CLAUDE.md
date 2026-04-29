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

All kwargs to `patch_model_for_lvp` are forwarded to every `Frozen*` wrapper via its `from_module` classmethod. String keys are resolved against `ACT_FN` / `BILINEAR_FN` at construction time.

```python
model = patch_model_for_lvp(
    model,
    # Attention softmax (Rule 3) — key into ACT_FN
    attn_act_fn='dtd_softmax',       # default; see Attention Softmax Variants below
    # QK and AV bilinear matmuls (Rule 4) — key into BILINEAR_FN
    matmul_fn='bilinear_matmul',     # default; use 'matmul' for standard torch.matmul
    # Gated-MLP gate×up product (Rule 4) — key into BILINEAR_FN
    mul_fn='bilinear_mul',           # default; use 'mul' for standard torch.mul
    # MLP activation (Rule 2) — key into ACT_FN; defaults to model-specific LVP fn
    mlp_act_fn='secant_silu',        # LLaMA/Qwen default; omit to use model default
    # Detach normalisation factor in RMSNorm / LayerNorm (Rule 1 linearisation)
    frozen_norm=True,                # default; set False to allow gradients through norm
    # Gemma2 only — softcap tanh (Rule 2) — key into ACT_FN
    attn_softcap_fn='secant_tanh',   # default
)
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
    └── patcher.py         # patch_model_for_lvp(model, include, exclude, nnsight_wrapper, dry_run, **kwargs)
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
- **Norm linearization:** `FrozenRMSNorm`/`FrozenLayerNorm` compute the normalization factor under `float32`. When `frozen_norm=True` (default) the factor is detached, making the layer linear in `x` so standard autograd gives $\mathbf{W}^T \mathbf{t}$. Set `frozen_norm=False` to allow gradients to flow through the norm.
- **Attention softmax:** Controlled by `attn_act_fn` (default `'dtd_softmax'`). When `attn_weights.grad_fn` is `None` (no-grad context), the AV product falls back to plain `torch.matmul` regardless of `matmul_fn`.
- **Bilinear products:** `matmul_fn` (default `'bilinear_matmul'`) is used for both the QK and AV matmuls. `mul_fn` (default `'bilinear_mul'`) is used for the gate×up product in gated MLPs.
- **GQA:** `repeat_kv` expands K/V heads before attention (LLaMA/Gemma2).
- **Gemma2 softcapping:** logit softcap function is controlled by `attn_softcap_fn` (default `'secant_tanh'`, Rule 2).
- **Function lookup:** String keys are resolved via `ACT_FN` (activations) and `BILINEAR_FN` (bilinear ops) in `linear_transformer/modules/`. Both dicts are importable from `linear_transformer.modules`.
- **NNsight integration:** `patch_model_for_lvp` wraps the patched model in `NNsight` by default (`nnsight_wrapper=True`). Disable with `nnsight_wrapper=False`.
- **Extending to new models:** Call `register_lvp_module(MyHFClass, MyFrozenWrapper.from_module)` before `patch_model_for_lvp`.

## ACT_FN and BILINEAR_FN Lookup Tables

String keys passed as kwargs to `patch_model_for_lvp` are resolved via these dicts at `from_module` time.

### `ACT_FN` — `linear_transformer/modules/activations.py`

| Key | Function | Notes |
|-----|----------|-------|
| `'gelu'` | `F.gelu` | standard |
| `'gelu_tanh'` | `F.gelu(approximate='tanh')` | standard |
| `'silu'` | `F.silu` | standard |
| `'relu'` | `F.relu` | standard |
| `'tanh'` | `torch.tanh` | standard |
| `'softmax'` | `F.softmax` | standard |
| `'secant_gelu'` | `SecantGELU` | Rule 2 — erf GELU |
| `'secant_gelu_tanh'` | `SecantGELUTanh` | Rule 2 — tanh-approx GELU (GPT-2, Gemma2 MLP default) |
| `'secant_silu'` | `SecantSiLU` | Rule 2 — SiLU/Swish (LLaMA/Qwen MLP default) |
| `'secant_relu'` | `SecantReLU` | Rule 2 — ReLU |
| `'secant_tanh'` | `SecantTanh` | Rule 2 — tanh (Gemma2 softcap default) |
| `'dtd_softmax'` | `DTDSoftmax` | Rule 3 DTD — $\mathbf{t} - \mathbf{s}(\mathbf{s}^T\mathbf{t})$, conservation-preserving (attn default) |
| `'sec_jac_softmax'` | `SecantJacobianSoftmax` | Rule 3 analytic secant-Jacobian |
| `'integrated_softmax'` | `IntegratedSoftmax` | Rule 3 integrated gradients (10 steps) |
| `'secant_softmax'` | `SecantSoftmax` | element-wise secant — $\mathbf{s}/(x+\epsilon) \cdot \sum t_j$ |
| `'pos_ratio_softmax'` | `PosRationSoftmax` | clips x to positive, uniform split |
| `'constant_softmax'` | detached `F.softmax` | treats attention weights as a constant |

### `BILINEAR_FN` — `linear_transformer/modules/bilinear.py`

| Key | Function | Notes |
|-----|----------|-------|
| `'mul'` | `torch.mul` | standard element-wise product |
| `'matmul'` | `torch.matmul` | standard matmul |
| `'bilinear_mul'` | `BilinearMul` | Rule 4 — uniform split, gate×up default |
| `'bilinear_matmul'` | `BilinearMatmul` | Rule 4 — uniform split, QK/AV default |
