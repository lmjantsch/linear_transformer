from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from transformers.cache_utils import Cache 
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.gemma2.configuration_gemma2 import Gemma2Config
from transformers.models.gemma2.modeling_gemma2 import repeat_kv, apply_rotary_pos_emb

from linear_transformer.modules import ACT_FN, BILINEAR_FN
from linear_transformer.modules.activations import dtd_softmax, secant_tanh
from linear_transformer.modules.bilinear import bilinear_mul, bilinear_matmul

class FrozenGemma2RMSNorm(nn.Module):

    def __init__(self, weight: nn.Parameter, eps: float, frozen_norm: bool) -> None:
        super().__init__()
        self.weight = weight
        self.eps = eps
        self.frozen_norm = frozen_norm

    @classmethod
    def from_module(cls, m: nn.Module, **kwargs: dict | None) -> FrozenGemma2RMSNorm:
        return cls(weight=m.weight, eps=m.eps, frozen_norm=kwargs.get('frozen_norm', False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, N, d_model)
        x_f = x.float()
        rms = x_f.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()  # (B, N, 1)
        w = (1.0 + self.weight.float()) / rms
        if self.frozen_norm:
            w = w.detach()  # (B, N, d_model) — effective linear weight, detach to freeze

        out = (x_f * w).to(x.dtype)
        return out

class FrozenGemma2GeGLU(nn.Module):

    def __init__(self, gate_proj: nn.Linear, up_proj: nn.Linear, down_proj: nn.Linear, act_fn: callable, mul_fn: callable) -> None:
        super().__init__()
        self.gate_proj = gate_proj   # (d_ffn, d_model)
        self.up_proj = up_proj       # (d_ffn, d_model)
        self.down_proj = down_proj   # (d_model, d_ffn)
        self.act_fn = act_fn
        self.mul_fn = mul_fn

    @classmethod
    def from_module(cls, m: nn.Module, **kwargs: dict | None) -> FrozenGemma2GeGLU:
        return cls(
            gate_proj=m.gate_proj,
            up_proj=m.up_proj,
            down_proj=m.down_proj,
            act_fn=ACT_FN[kwargs.get('mlp_act_fn', 'gelu_tanh')],
            mul_fn=BILINEAR_FN[kwargs.get('mul_fn', 'mul')],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, N, d_model)
        gate_pre = self.gate_proj(x)
        gate = self.act_fn(gate_pre)
        up = self.up_proj(x)
        hidden = self.mul_fn(gate, up)
        out = self.down_proj(hidden)
        return out

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float | int = 0.0,
    scaling: float | None = None,
    softcap: float | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scaling is None:
        scaling = module.head_dim**-0.5

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = module.matmul_fn(query, key_states.transpose(2, 3)) * scaling

    if softcap is not None:
        attn_weights = attn_weights / softcap
        attn_weights = module.attn_softcap_fn(attn_weights)
        attn_weights = attn_weights * softcap
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = module.attn_act_fn(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    # attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    if not attn_weights.grad_fn: # attention is dealt as constant if not part of the gradient graph
        attn_output = torch.matmul(attn_weights, value_states)
    else:
        attn_output = module.matmul_fn(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights

class FrozenGemma2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Gemma2Config, layer_idx: int, q_proj: nn.Linear, k_proj: nn.Linear, v_proj: nn.Linear, o_proj: nn.Linear,
                 attn_act_fn: callable, matmul_fn: callable, attn_softcap_fn: callable, **kwargs):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = config.query_pre_attn_scalar**-0.5
        self.attention_dropout = self.config.attention_dropout
        self.is_causal = not getattr(config, "use_bidirectional_attention", False)

        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.o_proj = o_proj
        self.attn_logit_softcapping = self.config.attn_logit_softcapping
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None

        self.attn_act_fn = attn_act_fn
        self.matmul_fn = matmul_fn
        self.attn_softcap_fn = attn_softcap_fn

    @classmethod
    def from_module(cls, m: nn.Module, **kwargs: dict | None) -> FrozenGemma2Attention:
        return cls(
            config=m.config,
            layer_idx=m.layer_idx,
            q_proj=m.q_proj,
            k_proj=m.k_proj,
            v_proj=m.v_proj,
            o_proj=m.o_proj,
            attn_act_fn=ACT_FN[kwargs.get('attn_act_fn', 'softmax')],
            matmul_fn=BILINEAR_FN[kwargs.get('matmul_fn', 'matmul')],
            attn_softcap_fn=ACT_FN[kwargs.get('attn_softcap_fn', 'tanh')],
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,

        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        # keep attention interface to unify nnsight self_attn.source tracing
        attention_interface = eager_attention_forward

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            softcap=self.attn_logit_softcapping,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights