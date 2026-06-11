# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install in editable mode (run once from repo root)
pip install -e ".[dev]"

# Run all tests
pytest

# Run only module-level tests (fast, no model downloads)
pytest tests/modules/

# Run model identity tests (downloads gpt2 on first run)
pytest tests/models/

# Run a single test file
pytest tests/modules/test_activations.py

# Run with extended per-layer diagnostics
pytest tests/models/ --extended

# Override device, model, or patching kwargs
pytest tests/ --device cpu --model gpt2 --attn_act_fn softmax --mul_fn bilinear_mul
```

## Architecture

This library lets you swap the backward pass of a HuggingFace transformer in-place, replacing standard `nn.Module` subgraphs with `Modular*` wrappers that implement Layer-wise Vector Propagation (LVP) — a gradient-based attribution method that modifies only the backward pass while keeping the forward pass numerically identical.

### Patching pipeline (`src/modular_transformer/patching/`)

`patch_model(model, **kwargs)` is the main entry point. It walks the module tree, looks up each `nn.Module` subclass in the registry, and swaps it for the corresponding `Modular*` wrapper via `from_module`. After patching, the model is optionally wrapped in `nnsight.NNsight` for activation access.

`registry.py` maps HuggingFace module types (e.g. `LlamaRMSNorm`, `GPT2MLP`) to their `Modular*` factory callables. HF imports are deferred so missing optional dependencies only raise when the registry is first used, not at import time. Custom modules can be added via `register_module(hf_class, factory)`.

### Module wrappers (`src/modular_transformer/models/`)

Each model subdirectory (`llama2/`, `gemma2/`, `gpt2/`, `generic/`) contains `Modular*` subclasses of `ModularModule` (defined in `base.py`). Every wrapper:
- Shares the original parameters (no copies) to preserve forward-pass identity.
- Implements `from_module(cls, m, **kwargs)` to construct itself from the original module.
- Selects its activation/norm/bilinear ops from the `ACT_FN`, `NORM_FN`, and `BILINEAR_FN` dicts using string kwargs passed through `patch_model`.

`base.py` also defines `ArchAccessors`, a dataclass of callables used to navigate a model's layer structure (e.g. `layers`, `attn`, `mlp`, `q_proj`). This is separate from the patching machinery.

### LVP operations (`src/modular_transformer/modules/`)

All operations that need a custom backward pass are implemented as `torch.autograd.Function` subclasses:

- **`activations.py`** — `SecantGELUTanh`, `SecantSiLU` (Rule 2: zero-preserving secant), `IntegratedSoftmax`, `SecantJacobianSoftmax` (Rule 3: softmax with principled backward). Also contains norm nonlinearities: `rms_norm`, `frozen_rms_norm`, `layer_norm`, `frozen_layer_norm` (frozen variants detach the denominator/centering term).
- **`bilinear.py`** — `BilinearMul`, `BilinearMatmul` (Rule 4: uniform 50/50 backward split), `DynamicBilinearMul`, `DynamicBilinearMatmul` (integrated-gradients variant using a midpoint between input and baseline).

Each operation also has a `@modular_ctx`-decorated functional wrapper (e.g. `secant_silu`, `bilinear_matmul`) that is stored in the `ACT_FN` / `NORM_FN` / `BILINEAR_FN` dicts and selected by name at patch time. `@modular_ctx` (`modules/utils.py`) is a thin decorator that allows a future forward-context hook to inject extra kwargs; currently it is a no-op pass-through.

### `patch_model` kwargs

These string keys are forwarded from `patch_model(**kwargs)` to each `from_module` call:

| kwarg | choices | effect |
|---|---|---|
| `norm_approx` | `"frozen"` / default | detach norm denominator in backward |
| `attn_act_fn` | `"softmax"`, `"sec_jac_softmax"`, `"integrated_softmax"` | softmax backward rule |
| `matmul_fn` | `"matmul"`, `"bilinear_matmul"`, `"ig_matmul"` | QK / AV product backward |
| `mul_fn` | `"mul"`, `"bilinear_mul"`, `"ig_mul"` | gated-MLP element-wise product backward |
| `mlp_act_fn` | `"silu"`, `"secant_silu"`, `"gelu_tanh"`, `"secant_gelu_tanh"`, … | MLP activation backward |

### Tests

- `tests/modules/` — unit tests for individual `autograd.Function` classes. Key properties: forward identity with `F.*` equivalents, gradient shape, and LVP conservation (`(x · ∇x).sum()` preserved through the op).
- `tests/models/` — integration tests that load real HF models, patch them, run a forward+backward pass, and check: (1) hidden-state identity (`rel_l2 < 1e-2`), (2) KL divergence of logits (`< 5e-3`), (3) LVP conservation per-module (`< 0.1`), and (4) total attribution at the embedding layer matching the loss.

Model fixtures are session-scoped and clean up GPU memory via `gc.collect()` + `torch.cuda.empty_cache()` after each model.
