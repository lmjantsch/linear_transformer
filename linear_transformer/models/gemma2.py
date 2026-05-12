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
from linear_transformer.modules.activations import dtd_softmax, secant_tanh, SoftcapFN
from linear_transformer.modules.bilinear import bilinear_mul, bilinear_matmul
from linear_transformer.models.utils import baseline_hidden_hook

class LinearGemma2RMSNorm(nn.Module):

    def __init__(self, weight: nn.Parameter, eps: float, norm_approx: str | None = None) -> None:
        super().__init__()
        self.weight = weight
        self.eps = eps
        self.norm_approx = norm_approx

    @classmethod
    def from_module(cls, m: nn.Module, **kwargs: dict | None) -> LinearGemma2RMSNorm:
        return cls(
            weight=m.weight,
            eps=m.eps,
            norm_approx=kwargs.get('norm_approx', None),
        )

    def forward(self, x):
        x_f = x.float()
        baseline_hidden = baseline_hidden_hook()
        dynamic_mask = baseline_hidden_hook()

        variance = x_f.pow(2).mean(-1, keepdim=True)

        if self.norm_approx in ('dynamic_thr', 'dynamic_msk') and (baseline_hidden is not None or dynamic_mask is not None):
            if self.norm_approx == 'dynamic_thr':
                baseline_hidden = baseline_hidden.to(torch.float32)
                clean_norm = x_f.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                scale = baseline_hidden.norm(dim=-1, keepdim=True) / clean_norm
                sim = torch.cosine_similarity(x_f, baseline_hidden, dim=-1).unsqueeze(-1)
                sin2 = (1 - sim.pow(2)).clamp(min=0)
                f_diff = (scale - 1).abs() - ((1 - sim).pow(2) + sin2 * (scale - 1).pow(2)).sqrt()
                dynamic_mask = f_diff >= 0.0
            rms_term = torch.rsqrt(variance + self.eps)
            rms_term = torch.where(dynamic_mask, rms_term, rms_term.detach())
            output = x_f * rms_term * (1.0 + self.weight.float()).detach()
            return output.type_as(x)

        # frozen
        if self.norm_approx == 'frozen':
            output = x_f * torch.rsqrt(variance + self.eps).detach() * (1.0 + self.weight.float()).detach()
            return output.type_as(x)
        # original
        output = x_f * torch.rsqrt(variance + self.eps) * (1.0 + self.weight.float())
        return output.type_as(x)


class LinearGemma2GeGLU(nn.Module):

    def __init__(self, gate_proj: nn.Linear, up_proj: nn.Linear, down_proj: nn.Linear, act_fn: callable, mul_fn: callable) -> None:
        super().__init__()
        self.gate_proj = gate_proj   # (d_ffn, d_model)
        self.up_proj = up_proj       # (d_ffn, d_model)
        self.down_proj = down_proj   # (d_model, d_ffn)
        self.act_fn = act_fn
        self.mul_fn = mul_fn

    @classmethod
    def from_module(cls, m: nn.Module, **kwargs: dict | None) -> LinearGemma2GeGLU:
        return cls(
            gate_proj=m.gate_proj,
            up_proj=m.up_proj,
            down_proj=m.down_proj,
            act_fn=ACT_FN[kwargs.get('mlp_act_fn', 'gelu_tanh')],
            mul_fn=BILINEAR_FN[kwargs.get('mul_fn', 'mul')],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, N, d_model)
        # original (replaced * with 'mul_fn)
        down_proj = self.down_proj(
            self.mul_fn(self.act_fn(self.gate_proj(x)), self.up_proj(x))
        )
        return down_proj

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

    # TODO Do I have to take care of this seperately? Bypass gradient
    # if softcap is not None:
    #     attn_weights = SoftcapFN.apply(attn_weights, softcap)
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = module.attn_act_fn(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    # attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = module.matmul_fn(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class LinearGemma2Attention(nn.Module):
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
    def from_module(cls, m: nn.Module, **kwargs: dict | None) -> LinearGemma2Attention:
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

        # hardcode attention_interface to 'eager', keep line for nnsight .source similarity with original model
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