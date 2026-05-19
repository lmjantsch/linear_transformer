from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.models.gemma2.configuration_gemma2 import Gemma2Config
from transformers.models.gemma2.modeling_gemma2 import repeat_kv, apply_rotary_pos_emb

from linear_transformer.modules import ACT_FN, BILINEAR_FN
from linear_transformer.models.utils import CustomModule, hidden_junction_hook, gradient_junction_hook

class CustomGemma2RMSNorm(CustomModule):

    def __init__(self, weight: nn.Parameter, eps: float, norm_approx: str | None = None) -> None:
        super().__init__()
        self.weight = weight
        self.eps = eps
        self.norm_approx = norm_approx

    @classmethod
    def from_module(cls, m: nn.Module, **kwargs) -> CustomGemma2RMSNorm:
        return cls(
            weight=m.weight,
            eps=m.eps,
            norm_approx=kwargs.get('norm_approx', None),
        )

    def forward(self, x: torch.Tensor, head_wise: bool = False) -> torch.Tensor:
        x_f = x.float()
        weight = 1.0 + self.weight.float()

        if head_wise:
            # RMS from sum of heads so all heads share the same scale factor:
            # sum(h_i * rms_inv) == sum(h_i) * rms_inv  (B, S, H, d_model)
            variance = x_f.sum(dim=2).pow(2).mean(-1, keepdim=True)   # (B, S, 1)
            rms_term = torch.rsqrt(variance + self.eps).unsqueeze(2)   # (B, S, 1, 1)
        else:
            variance = x_f.pow(2).mean(-1, keepdim=True)
            rms_term = torch.rsqrt(variance + self.eps)

        if self.norm_approx == 'frozen':
            output = x_f * rms_term.detach() * weight.detach()
        else:
            output = x_f * rms_term * weight
        return output.type_as(x)


class CustomGemma2MLP(CustomModule):

    def __init__(self, gate_proj: nn.Linear, up_proj: nn.Linear, down_proj: nn.Linear, act_fn: callable, mul_fn: callable) -> None:
        super().__init__()
        self.gate_proj = gate_proj   # (d_ffn, d_model)
        self.up_proj = up_proj       # (d_ffn, d_model)
        self.down_proj = down_proj   # (d_model, d_ffn)
        self.act_fn = act_fn
        self.mul_fn = mul_fn

    @classmethod
    def from_module(cls, m: nn.Module, **kwargs) -> CustomGemma2MLP:
        return cls(
            gate_proj=m.gate_proj,
            up_proj=m.up_proj,
            down_proj=m.down_proj,
            act_fn=ACT_FN[kwargs.get('mlp_act_fn', 'gelu_tanh')],
            mul_fn=BILINEAR_FN[kwargs.get('mul_fn', 'mul')],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, N, d_model)
        _, gate_x, up_x = gradient_junction_hook(x)

        gate_out = self.gate_proj(gate_x)
        gate_act = self.act_fn(gate_out)
        up_out = self.up_proj(up_x)

        z = self.mul_fn(gate_act, up_out)
        down_out = self.down_proj(z)
        return down_out

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

    # key_states = repeat_kv(key, module.num_key_value_groups)
    # value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = module.matmul_fn(query, key.transpose(2, 3)) * scaling

    if softcap is not None:
        attn_weights = attn_weights / softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * softcap
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = module.attn_act_fn(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    # attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = module.matmul_fn(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class CustomGemma2Attention(CustomModule):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Gemma2Config, layer_idx: int, q_proj: nn.Linear, k_proj: nn.Linear, v_proj: nn.Linear, o_proj: nn.Linear,
                 attn_act_fn: callable, matmul_fn: callable, **kwargs):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.config = config
        self.layer_idx = layer_idx
        self.n_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = config.query_pre_attn_scalar**-0.5
        self.attention_dropout = self.config.attention_dropout
        self.is_causal = not getattr(config, "use_bidirectional_attention", False)

        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.o_proj = o_proj
        self.attn_logit_softcapping = self.config.attn_logit_softcapping # create handle to deactivate
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None

        self.attn_act_fn = attn_act_fn
        self.matmul_fn = matmul_fn

    @classmethod
    def from_module(cls, m: nn.Module, **kwargs) -> CustomGemma2Attention:
        return cls(
            config=m.config,
            layer_idx=m.layer_idx,
            q_proj=m.q_proj,
            k_proj=m.k_proj,
            v_proj=m.v_proj,
            o_proj=m.o_proj,
            attn_act_fn=ACT_FN[kwargs.get('attn_act_fn', 'softmax')],
            matmul_fn=BILINEAR_FN[kwargs.get('matmul_fn', 'matmul')],
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,

        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        _, B, S, D = hidden_states.shape
        q_hidden, k_hidden, v_hidden = gradient_junction_hook(hidden_states)[1:].view(3, self.n_heads, B, S, D)
        input_shape = (B, S)
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self._head_wise_projection(q_hidden, self.q_proj).view(hidden_shape).transpose(1, 2)
        key_states = self._head_wise_projection(k_hidden, self.k_proj, is_grouped=True).view(hidden_shape).transpose(1, 2)
        value_states = self._head_wise_projection(v_hidden, self.v_proj, is_grouped=True).view(hidden_shape).transpose(1, 2)

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

        # head-wise attention output
        head_wise_attn_output = torch.einsum('BSHd, DHd -> HBSD', attn_output.detach(), self.o_proj.weight.reshape(-1, self.n_heads, self.head_dim))
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        attn_output = torch.cat([attn_output.unsqueeze(0), head_wise_attn_output])
        return attn_output, attn_weights
    
    def _head_wise_projection(self, hidden_state: torch.Tensor, proj: nn.Linear, is_grouped:bool = False) -> torch.Tensor:
        _, B, S, D = hidden_state.shape
        
        if is_grouped:
            hidden_state = hidden_state.view(self.num_key_value_groups, self.config.num_key_value_heads, B, S, D)
            weight_view = proj.weight.view(self.config.num_key_value_heads, self.head_dim, D)
            out = torch.einsum("GHBSD, HdD -> BSGHd", hidden_state, weight_view)

            if proj.bias is not None:
                out += proj.bias.view(self.config.num_key_value_heads, self.head_dim)
            return out.reshape(B, S, self.n_heads, self.head_dim)

        weight_view = proj.weight.view(self.n_heads, self.head_dim, D)
        out = torch.einsum("HBSD, HdD -> BSHd", hidden_state, weight_view)

        if proj.bias is not None:
            out += proj.bias.view(self.n_heads, self.head_dim)
        return out.contiguous()
    
class CustomGemma2DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, hidden_size: int, self_attn: nn.Module, mlp: nn.Module,
                 input_layernorm: nn.Module, post_attention_layernorm: nn.Module,
                 pre_feedforward_layernorm: nn.Module, post_feedforward_layernorm: nn.Module):
        super().__init__()
        self.hidden_size = hidden_size
        self.self_attn = self_attn
        self.mlp = mlp
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm
        self.pre_feedforward_layernorm = pre_feedforward_layernorm
        self.post_feedforward_layernorm = post_feedforward_layernorm

    @classmethod
    def from_module(cls, m: nn.Module, **kwargs) -> CustomGemma2DecoderLayer:
        return cls(
            hidden_size=m.hidden_size,
            self_attn=m.self_attn,
            mlp=m.mlp,
            input_layernorm=m.input_layernorm,
            post_attention_layernorm=m.post_attention_layernorm,
            pre_feedforward_layernorm=m.pre_feedforward_layernorm,
            post_feedforward_layernorm=m.post_feedforward_layernorm,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hqkv_states = hidden_junction_hook(hidden_states, n=(3 * self.self_attn.n_heads))  # (1 + 3 * H, B, N, d_model)
        hqkv_states = self.input_layernorm(hqkv_states)

        # Self Attention
        attn_output, _ = self.self_attn(
            hidden_states=hqkv_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        attn_output = self.post_attention_layernorm(attn_output[0])  # (B, S, H, d_model)
        hidden_states = residual + attn_output

        # Fully Connected
        residual = hidden_states
        hgu_states = hidden_junction_hook(hidden_states, n=2)  # (3, B, N, d_model)
        hgu_states = self.pre_feedforward_layernorm(hgu_states)
        hidden_states = self.mlp(hgu_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states