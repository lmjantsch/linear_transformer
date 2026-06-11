from __future__ import annotations

import torch
import torch.nn as nn

from transformers.cache_utils import Cache

from modular_transformer.modules import ACT_FN, BILINEAR_FN
from modular_transformer.models.utils import conv1d_to_linear, split_c_attn
from modular_transformer.models.base import ModularModule


class ModularGPT2Attention(ModularModule):

    def __init__(
        self,
        config,
        layer_idx: int,
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
        o_proj: nn.Linear,
        attn_act_fn: callable,
        matmul_fn: callable,
        attention_interface: callable,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.is_causal = True

        self.scaling = 1.0
        if config.scale_attn_weights:
            self.scaling = self.head_dim ** -0.5
        if config.scale_attn_by_inverse_layer_idx:
            self.scaling /= float(layer_idx + 1)

        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.o_proj = o_proj
        self.attn_act_fn = attn_act_fn
        self.matmul_fn = matmul_fn
        self.attention_interface = attention_interface

    @classmethod
    def from_module(cls, m: nn.Module, **kwargs) -> ModularGPT2Attention:
        q_proj, k_proj, v_proj = split_c_attn(m.c_attn, m.embed_dim)
        return cls(
            config=m.config,
            layer_idx=m.layer_idx,
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            o_proj=conv1d_to_linear(m.c_proj),
            attn_act_fn=ACT_FN[kwargs.get('attn_act_fn', 'softmax')],
            matmul_fn=BILINEAR_FN[kwargs.get('matmul_fn', 'matmul')],
            attention_interface=ATTN_INTERFACE_FN[kwargs.get('attn_interface', 'eager')],
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # (B, H, N, d_head)
        key_states   = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attn_output, attn_weights = self.attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


def eager_attention_forward(
    module: ModularGPT2Attention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scaling is None:
        scaling = query.size(-1) ** -0.5

    attn_weights = module.matmul_fn(query, key.transpose(-1, -2)) * scaling  # (B, H, N, N)

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = module.attn_act_fn(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    attn_output = module.matmul_fn(attn_weights, value)
    attn_output = attn_output.transpose(1, 2)
    return attn_output, attn_weights


ATTN_INTERFACE_FN = {
    'eager': eager_attention_forward,
}
