from __future__ import annotations

from typing import Any

import torch
from torch import nn

from transformers.integrations import use_kernelized_func
from transformers.cache_utils import Cache 
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import repeat_kv, apply_rotary_pos_emb

from linear_transformer.modules.activations import secant_silu, dtd_softmax
from linear_transformer.modules.bilinear import bilinear_mul, bilinear_matmul

class FrozenLlama2RMSNorm(nn.Module):

    def __init__(self, weight: nn.Parameter, eps: float) -> None:
        super().__init__()
        self.weight = weight
        self.eps = eps

    @classmethod
    def from_module(cls, m: nn.Module, **kwards: dict | None) -> FrozenLlama2RMSNorm:
        return cls(weight=m.weight, eps=m.variance_epsilon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, N, d_model)
        x_f = x.float()
        rms = x_f.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt().detach()  # (B, N, 1) detache to freeze                                                                                                                                                                                                                                                                                 
        out = self.weight.detach() * (x_f / rms).to(x.dtype) 
        return out

class FrozenLlama2SwiGLU(nn.Module):

    def __init__(self, gate_proj: nn.Linear, up_proj: nn.Linear, down_proj: nn.Linear, **kwargs) -> None:
        super().__init__()
        self.gate_proj = gate_proj   # (d_ffn, d_model)
        self.up_proj = up_proj       # (d_ffn, d_model)
        self.down_proj = down_proj   # (d_model, d_ffn)

    @classmethod
    def from_module(cls, m: nn.Module, **kwargs: dict | None) -> FrozenLlama2SwiGLU:
        return cls(gate_proj=m.gate_proj, up_proj=m.up_proj, down_proj=m.down_proj, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, N, d_model)
        gate_pre = self.gate_proj(x)
        gate = secant_silu(gate_pre)
        up = self.up_proj(x)
        hidden = bilinear_mul(gate, up)                   
        out = self.down_proj(hidden)   
        return out
    
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    act_fn: callable,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = bilinear_matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = act_fn(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    # attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    if not attn_weights.grad_fn: # attention is dealt as constant if not part of the gradient graph
        attn_output = torch.matmul(attn_weights, value_states)
    else:
        attn_output = bilinear_matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

@use_kernelized_func(apply_rotary_pos_emb)
class FrozenLlama2Attention(nn.Module):
    """Llama2 attention, largely unchanged"""

    def __init__(self, config: LlamaConfig, layer_idx: int, q_proj: nn.Linear, k_proj: nn.Linear, v_proj: nn.Linear, o_proj: nn.Linear, **kwargs):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.o_proj = o_proj
        self.attn_act_fn = kwargs.get('attn_act_fn', dtd_softmax)

    @classmethod
    def from_module(cls, m: nn.Module, **kwargs: dict | None) -> FrozenLlama2Attention:
        return cls(
            config = m.config,
            layer_idx = m.layer_idx,
            q_proj = m.q_proj,
            k_proj = m.k_proj,
            v_proj = m.v_proj,
            o_proj = m.o_proj,
            **kwargs
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            act_fn=self.attn_act_fn,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights