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

Rule 1 (linear layers) and norm layers are handled structurally: `Modular*` wrappers detach the normalization factor — standard autograd then produces $\mathbf{W}^T \mathbf{t}$ automatically.

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

All kwargs to `patch_model_for_lvp` are forwarded to every `Modular*` wrapper via its `from_module` classmethod. String keys are resolved against `ACT_FN` / `BILINEAR_FN` at construction time.

```python
model = patch_model_for_lvp(
    model,
    # Attention softmax (Rule 3) — key into ACT_FN
    attn_act_fn='dtd_softmax',       # default 'softmax'; see Attention Softmax Variants below
    # QK and AV bilinear matmuls (Rule 4) — key into BILINEAR_FN
    matmul_fn='bilinear_matmul',     # default 'matmul' (standard torch.matmul)
    # Gated-MLP gate×up product (Rule 4) — key into BILINEAR_FN
    mul_fn='bilinear_mul',           # default 'mul' (standard torch.mul)
    # MLP activation (Rule 2) — key into ACT_FN; defaults to model-specific standard fn
    mlp_act_fn='secant_silu',        # LLaMA/Qwen; 'secant_gelu_tanh' for GPT-2/Gemma2
    # Detach normalisation factor in RMSNorm / LayerNorm (Rule 1 linearisation)
    norm_approx='frozen',            # default None (standard norm); 'frozen' detaches norm factor
    # Attention interface — key into each model's ATTN_INTERFACE_FN
    attn_interface='eager',          # default; currently only 'eager' is supported
)
```

## Code Structure

```
linear_transformer/
├── __init__.py            # exports patch_model_for_lvp, register_lvp_module
├── modules/
│   ├── activations.py     # Rule 2/3 autograd.Function nodes + functional wrappers + ACT_FN
│   └── bilinear.py        # BilinearMul, BilinearMatmul (Rule 4) + BILINEAR_FN
├── models/
│   ├── utils.py           # expand_kv_linear, conv1d_to_linear, split_c_attn
│   ├── generic/
│   │   └── layernorm.py   # ModularLayerNorm
│   ├── gpt2/
│   │   ├── gpt2_attention.py  # ModularGPT2Attention
│   │   └── gpt2_mlp.py        # ModularGPT2MLP
│   ├── llama2/
│   │   ├── llama2_attention.py  # ModularLlama2Attention
│   │   ├── llama2_mlp.py        # ModularLlama2MLP
│   │   └── llama2_rmsnorm.py    # ModularLlama2RMSNorm
│   └── gemma2/
│       ├── gemma2_attention.py  # ModularGemma2Attention
│       ├── gemma2_mlp.py        # ModularGemma2MLP
│       └── gemma2_rmsnorm.py    # ModularGemma2RMSNorm
└── patching/
    ├── registry.py        # _REGISTRY; register_lvp_module(); get_registry()
    └── patcher.py         # patch_model_for_lvp(model, include, exclude, nnsight_wrapper, dry_run, **kwargs)
```

## Supported Models

| Model | Attn wrapper | MLP wrapper | Norm wrapper |
|-------|-------------|------------|-------------|
| GPT-2 | `ModularGPT2Attention` (no RoPE, c_attn split to q/k/v) | `ModularGPT2MLP` (tanh-GELU) | `ModularLayerNorm` |
| LLaMA 3.1 | `ModularLlama2Attention` (RoPE + GQA via expand_kv_linear) | `ModularLlama2MLP` (SiLU gate) | `ModularLlama2RMSNorm` |
| Qwen2.5 | `ModularLlama2Attention` (RoPE + GQA via expand_kv_linear) | `ModularLlama2MLP` (SiLU gate) | `ModularLlama2RMSNorm` |
| Gemma2 | `ModularGemma2Attention` (RoPE + GQA via expand_kv_linear + softcap) | `ModularGemma2MLP` (tanh-GELU gate) | `ModularGemma2RMSNorm` |

## Key Implementation Details

- **Weight sharing:** `from_module(m)` stores references to original parameters — no weight copies, zero memory overhead. Exception: KV projections in GQA models are expanded via `expand_kv_linear` (new tensor).
- **Precision:** All rule computations cast to float32 internally, cast back to original dtype on return.
- **Norm linearization:** Each norm module accepts `norm_approx` kwarg. `None` (default) runs the standard norm. `'frozen'` detaches the normalization factor and weight, making the layer linear in `x` so standard autograd gives $\mathbf{W}^T \mathbf{t}$.
- **GQA:** K/V projections are pre-expanded via `expand_kv_linear` in `from_module`, replacing `repeat_kv` at forward time. The expanded projection has shape `(num_heads * d_head, d_model)`.
- **GPT-2 Conv1D:** `split_c_attn` splits the fused `c_attn` Conv1D into separate `q_proj`, `k_proj`, `v_proj` nn.Linear layers. `conv1d_to_linear` converts a single Conv1D (transposes the weight).
- **Attention softmax:** Controlled by `attn_act_fn` (default `'softmax'`).
- **Bilinear products:** `matmul_fn` (default `'matmul'`) is used for QK and AV matmuls. `mul_fn` (default `'mul'`) is used for the gate×up product in gated MLPs.
- **Gemma2 softcapping:** The logit softcap uses standard `torch.tanh` (hardcoded in `eager_attention_forward`).
- **Attention interface:** Each model's attention module has an `ATTN_INTERFACE_FN` dict. Currently only `'eager'` is supported; selected via `attn_interface` kwarg.
- **Function lookup:** String keys are resolved via `ACT_FN` (activations) and `BILINEAR_FN` (bilinear ops) in `linear_transformer/modules/`. Both dicts are importable from `linear_transformer.modules`.
- **NNsight integration:** `patch_model_for_lvp` wraps the patched model in `NNsight` by default (`nnsight_wrapper=True`). Disable with `nnsight_wrapper=False`.
- **Extending to new models:** Call `register_lvp_module(MyHFClass, MyModularWrapper.from_module)` before `patch_model_for_lvp`.

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
| `'secant_tanh'` | `SecantTanh` | Rule 2 — tanh |
| `'dtd_softmax'` | `DTDSoftmax` | Rule 3 DTD — $\mathbf{t} - \mathbf{s}(\mathbf{s}^T\mathbf{t})$, conservation-preserving |
| `'sec_jac_softmax'` | `SecantJacobianSoftmax` | Rule 3 analytic secant-Jacobian |
| `'integrated_softmax'` | `IntegratedSoftmax` | Rule 3 integrated gradients (10 steps) |
| `'secant_softmax'` | `SecantSoftmax` | element-wise secant — $\mathbf{s}/(x+\epsilon) \cdot \sum t_j$ |
| `'pos_ratio_softmax'` | `PosRationSoftmax` | clips x to positive, uniform split |
| `'outer_prod_softmax'` | `OuterProdSoftmax` | outer-product projection onto x |
| `'frozen_denom_softmax'` | `FrozenDenomSoftmax` | detached denominator |
| `'constant_softmax'` | detached `F.softmax` | treats attention weights as a constant |

### `BILINEAR_FN` — `linear_transformer/modules/bilinear.py`

| Key | Function | Notes |
|-----|----------|-------|
| `'mul'` | `torch.mul` | standard element-wise product |
| `'matmul'` | `torch.matmul` | standard matmul |
| `'bilinear_mul'` | `BilinearMul` | Rule 4 — uniform split, gate×up |
| `'bilinear_matmul'` | `BilinearMatmul` | Rule 4 — uniform split, QK/AV |
