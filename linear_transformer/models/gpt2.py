from __future__ import annotations

import torch
import torch.nn as nn

from transformers.pytorch_utils import Conv1D
from transformers.cache_utils import Cache, EncoderDecoderCache
from transformers.utils.generic import maybe_autocast

from linear_transformer.modules import ACT_FN, BILINEAR_FN
from linear_transformer.modules.activations import secant_gelu_tanh, dtd_softmax
from linear_transformer.modules.bilinear import bilinear_matmul

class FrozenGPT2MLP(nn.Module):

    def __init__(self, c_fc: nn.Linear, c_proj: nn.Linear, act_fn=secant_gelu_tanh) -> None:
        super().__init__()
        self.c_fc = c_fc    # (d_model, d_ffn)
        self.c_proj = c_proj  # (d_ffn, d_model)
        self.act_fn = act_fn

    @classmethod
    def from_module(cls, m: nn.Module, **kwargs: dict | None) -> FrozenGPT2MLP:
        # GPT-2 MLP uses c_fc and c_proj (Conv1D, weight transposed vs nn.Linear)
        return cls(
            c_fc=m.c_fc,
            c_proj=m.c_proj,
            act_fn=ACT_FN[kwargs.get('mlp_act_fn', 'secant_gelu_tanh')],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, N, d_model)
        h = self.c_fc(x)
        h_act = self.act_fn(h)
        out = self.c_proj(h_act)
        return out
    

def eager_attention_forward(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    if scaling is None:
        scaling = query.size(-1) ** -0.5

    attn_weights = module.matmul_fn(query, key.transpose(-1, -2)) * scaling

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = module.attn_act_fn(attn_weights, dim=-1, dtype=torch.float32)

    # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
    attn_weights = attn_weights.type(value.dtype)
    # attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    if not attn_weights.grad_fn: # attention is dealt as constant if not part of the gradient graph
        attn_output = torch.matmul(attn_weights, value)
    else:
        attn_output = module.matmul_fn(attn_weights, value)
    attn_output = attn_output.transpose(1, 2)

    return attn_output, attn_weights


class FrozenGPT2Attention(nn.Module):
    def __init__(self, config, c_attn, q_attn, c_proj, is_cross_attention=False, layer_idx=None, **kwargs):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn
        self.is_cross_attention = is_cross_attention
        self.layer_idx = layer_idx

        # Precompute unified scaling factor (accounts for both head_dim and layer-wise scaling)
        self.scaling = 1.0
        if self.scale_attn_weights:
            self.scaling = self.head_dim**-0.5
        if self.scale_attn_by_inverse_layer_idx:
            self.scaling /= float(self.layer_idx + 1)

        self.q_attn = q_attn
        self.c_attn = c_attn
        self.c_proj = c_proj

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.is_causal = not is_cross_attention
        self.attn_act_fn = kwargs.get('attn_act_fn', dtd_softmax)
        self.matmul_fn = kwargs.get('matmul_fn', bilinear_matmul)

    @classmethod
    def from_module(cls, m: nn.Module, **kwargs: dict | None) -> FrozenGPT2Attention:
        return cls(
            config=m.config,
            is_cross_attention=m.is_cross_attention,
            layer_idx=m.layer_idx,
            c_attn=m.c_attn,
            q_attn=m.q_attn if hasattr(m, 'q_attn') else None,
            c_proj=m.c_proj,
            attn_act_fn=ACT_FN[kwargs.get('attn_act_fn', 'dtd_softmax')],
            matmul_fn=BILINEAR_FN[kwargs.get('matmul_fn', 'bilinear_matmul')],
        )

    def forward(
        self,
        hidden_states: tuple[torch.FloatTensor] | None,
        past_key_values: Cache | None = None,
        attention_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        output_attentions: bool | None = False,
        **kwargs,
    ) -> tuple[torch.Tensor | tuple[torch.Tensor], ...]:
        is_cross_attention = encoder_hidden_states is not None
        if past_key_values is not None:
            if isinstance(past_key_values, EncoderDecoderCache):
                is_updated = past_key_values.is_updated.get(self.layer_idx)
                if is_cross_attention:
                    # after the first generated id, we can subsequently re-use all key/value_layer from cache
                    curr_past_key_values = past_key_values.cross_attention_cache
                else:
                    curr_past_key_values = past_key_values.self_attention_cache
            else:
                curr_past_key_values = past_key_values

        if is_cross_attention:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )
            query_states = self.q_attn(hidden_states)
            attention_mask = encoder_attention_mask

            # Try to get key/value states from cache if possible
            if past_key_values is not None and is_updated:
                key_states = curr_past_key_values.layers[self.layer_idx].keys
                value_states = curr_past_key_values.layers[self.layer_idx].values
            else:
                key_states, value_states = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
                shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
                key_states = key_states.view(shape_kv).transpose(1, 2)
                value_states = value_states.view(shape_kv).transpose(1, 2)
        else:
            query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
            shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
            key_states = key_states.view(shape_kv).transpose(1, 2)
            value_states = value_states.view(shape_kv).transpose(1, 2)

        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        query_states = query_states.view(shape_q).transpose(1, 2)

        if (past_key_values is not None and not is_cross_attention) or (
            past_key_values is not None and is_cross_attention and not is_updated
        ):
            key_states, value_states = curr_past_key_values.update(key_states, value_states, self.layer_idx)
            # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
            if is_cross_attention:
                past_key_values.is_updated[self.layer_idx] = True

        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attn_dropout.p if self.training else 0.0,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        attn_output = self.c_proj(attn_output)
        # attn_output = self.resid_dropout(attn_output)

        return attn_output, attn_weights