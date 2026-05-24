import torch
import torch.nn as nn

from transformers.pytorch_utils import Conv1D


from dataclasses import dataclass
from typing import Callable, Annotated, Any

MODEL_SCOPE = "model"
LAYER_SCOPE = "layer"

ModelCallable = Annotated[Callable[[Any], Any], MODEL_SCOPE]
LayerCallable = Annotated[Callable[[Any], Any], LAYER_SCOPE]


@dataclass(kw_only=True)
class ArchAccessors:
    """Layer accessors for different transformer architectures."""

    # configs
    n_layers: ModelCallable                  # (model) -> int
    n_heads: ModelCallable                   # (model) -> int
    model_dim: ModelCallable                 # (model) -> int
    head_dim: ModelCallable                  # (model) -> int
    uses_rotary_emb: ModelCallable = lambda model: False

    # model modules
    embed: ModelCallable                                                             # (model) -> embed module
    pos_emb: ModelCallable = lambda model: None           # (model) -> position embeding module
    rotary_emb: ModelCallable = lambda model: None    # (model) -> rotary embeding module
    layers: ModelCallable                                                            # (model) -> layer list
    ln: ModelCallable                                                           # (model) -> lm_head module
    lm_head: ModelCallable                                                           # (model) -> lm_head module

    # layer modules
    pre_ln1: LayerCallable                                                           # (layer) -> pre-attention norm module
    attn: LayerCallable                                                              # (layer) -> attention module
    post_ln1: LayerCallable = lambda model:None                         # (layer) -> post-attention norm module

    pre_ln2: LayerCallable                                                           # (layer) -> pre-MLP norm module
    mlp: LayerCallable                                                               # (layer) -> MLP module
    post_ln2: LayerCallable = lambda model: None                        # (layer) -> post-MLP norm module

    # (layer) -> attn submodule
    q_proj: LayerCallable
    k_proj: LayerCallable
    v_proj: LayerCallable
    o_proj: LayerCallable
    attention_interface: LayerCallable

    # (layer) -> mlp submodule
    up_proj: LayerCallable
    gate_proj: LayerCallable = lambda layer: None
    down_proj: LayerCallable


def expand_kv_linear(
    linear: nn.Linear,
    num_kv_groups: int,
    d_head: int,
) -> nn.Linear:
    """Return a new Linear whose weights mirror repeat_kv applied to `linear`.

    Expands (kv_heads * d_head, d_model) → (num_heads * d_head, d_model) by
    repeating each head-block `num_kv_groups` times along the output dimension.
    """
    if num_kv_groups == 1:
        return linear

    W = linear.weight.data          # (kv_heads * d_head, d_model)
    d_model = W.shape[1]
    kv_heads = W.shape[0] // d_head

    W_exp = (
        W.view(kv_heads, d_head, d_model)
         .repeat_interleave(num_kv_groups, dim=0)  # (num_heads, d_head, d_model)
         .reshape(kv_heads * num_kv_groups * d_head, d_model)
    )

    out = nn.Linear(
        d_model, W_exp.shape[0],
        bias=linear.bias is not None,
        device=W.device, dtype=W.dtype,
    )
    out.weight = nn.Parameter(W_exp, requires_grad=linear.weight.requires_grad)

    if linear.bias is not None:
        b_exp = (
            linear.bias.data.view(kv_heads, d_head)
            .repeat_interleave(num_kv_groups, dim=0)
            .reshape(-1)
        )
        out.bias = nn.Parameter(b_exp, requires_grad=linear.bias.requires_grad)

    return out


def conv1d_to_linear(conv1d: Conv1D) -> nn.Linear:
    """Convert a HuggingFace Conv1D to an equivalent nn.Linear.

    Conv1D computes x @ W + b with W of shape (in, out).
    nn.Linear computes x @ W.T + b with W of shape (out, in).
    """
    W = conv1d.weight.data      # (in_features, out_features)
    in_features, out_features = W.shape
    linear = nn.Linear(in_features, out_features, device=W.device, dtype=W.dtype)
    linear.weight = nn.Parameter(W.T.contiguous(), requires_grad=conv1d.weight.requires_grad)
    if conv1d.bias is not None:
        linear.bias = nn.Parameter(conv1d.bias.data.clone(), requires_grad=conv1d.bias.requires_grad)
    return linear


def split_c_attn(
    c_attn: Conv1D,
    embed_dim: int,
) -> tuple[nn.Linear, nn.Linear, nn.Linear]:
    """Split GPT2's fused c_attn Conv1D into q_proj, k_proj, v_proj nn.Linear layers.

    c_attn weight has shape (embed_dim, 3 * embed_dim) in Conv1D convention.
    Each returned Linear has shape (embed_dim, embed_dim) in nn.Linear convention.
    """
    W = c_attn.weight.data      # (embed_dim, 3 * embed_dim)
    has_bias = c_attn.bias is not None

    W_q, W_k, W_v = W.split(embed_dim, dim=1)  # each (embed_dim, embed_dim)

    b_q = b_k = b_v = None
    if has_bias:
        b_q, b_k, b_v = c_attn.bias.data.split(embed_dim)

    def make_linear(W_slice: torch.Tensor, b_slice: torch.Tensor | None) -> nn.Linear:
        lin = nn.Linear(embed_dim, embed_dim, device=W.device, dtype=W.dtype)
        lin.weight = nn.Parameter(W_slice.T.contiguous(), requires_grad=W.requires_grad)
        if has_bias:
            lin.bias = nn.Parameter(b_slice.clone(), requires_grad=c_attn.bias.requires_grad)
        return lin

    return make_linear(W_q, b_q), make_linear(W_k, b_k), make_linear(W_v, b_v)


def identity_anchor(hidden_state: torch.Tensor) -> torch.Tensor:
    return hidden_state