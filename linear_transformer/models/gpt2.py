from __future__ import annotations

import torch
import torch.nn as nn

from transformers.cache_utils import Cache, EncoderDecoderCache
from transformers.modeling_layers import GradientCheckpointingLayer

from linear_transformer.modules import ACT_FN, BILINEAR_FN

from linear_transformer.models.utils import CustomModule, conv1d_to_linear, slice_conv1d, hidden_junction_hook, gradient_junction_hook

class CustomGPT2MLP(CustomModule):

    def __init__(self, up_proj: nn.Linear, down_proj: nn.Linear, act_fn: callable) -> None:
        super().__init__()
        self.up_proj = up_proj    # (d_ffn, d_model)
        self.down_proj = down_proj  # (d_model, d_ffn)
        self.act_fn = act_fn

    @classmethod
    def from_module(cls, m: nn.Module, **kwargs) -> CustomGPT2MLP:

        return cls(
            up_proj=conv1d_to_linear(m.c_fc.weight, m.c_fc.bias),
            down_proj=conv1d_to_linear(m.c_proj.weight, m.c_proj.bias),
            act_fn=ACT_FN[kwargs.get('mlp_act_fn', 'gelu_tanh')],
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        _, up_states, _ = gradient_junction_hook(hidden_states)
        hidden_states = self.up_proj(up_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.down_proj(hidden_states)
        # hidden_states = self.dropout(hidden_states)
        return hidden_states
    

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

    attn_output = module.matmul_fn(attn_weights, value)
    attn_output = attn_output.transpose(1, 2)

    return attn_output, attn_weights


class CustomGPT2Attention(CustomModule):
    def __init__(self, config, q_proj: nn.Linear, k_proj: nn.Linear, v_proj: nn.Linear,
                 out_proj, attn_act_fn: callable, matmul_fn: callable,
                 layer_idx: int | None = None, **kwargs):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn
        self.layer_idx = layer_idx
        self.is_causal = True

        # Precompute unified scaling factor (accounts for both head_dim and layer-wise scaling)
        self.scaling = 1.0
        if self.scale_attn_weights:
            self.scaling = self.head_dim**-0.5
        if self.scale_attn_by_inverse_layer_idx:
            self.scaling /= float(self.layer_idx + 1)

        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.out_proj = out_proj

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.attn_act_fn = attn_act_fn
        self.matmul_fn = matmul_fn

    @classmethod
    def from_module(cls, m: nn.Module, **kwargs) -> CustomGPT2Attention:
        if m.is_cross_attention:
            raise NotImplementedError("Cross-attention is not supported.")
        if kwargs.get('center_writing_weights', False):
            m.c_proj.weight.data = m.c_proj.weight.data - m.c_proj.weight.data.mean(dim=1, keepdim=True)

        embed_dim = m.embed_dim

        return cls(
            config=m.config,
            layer_idx=m.layer_idx,
            q_proj=slice_conv1d(m.c_attn, s=slice(0, embed_dim)),
            k_proj=slice_conv1d(m.c_attn, s=slice(embed_dim, 2 * embed_dim)),
            v_proj=slice_conv1d(m.c_attn, s=slice(2 * embed_dim, None)),
            out_proj=conv1d_to_linear(m.c_proj.weight, m.c_proj.bias),
            attn_act_fn=ACT_FN[kwargs.get('attn_act_fn', 'softmax')],
            matmul_fn=BILINEAR_FN[kwargs.get('matmul_fn', 'matmul')],
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
        if encoder_hidden_states is not None:
            raise NotImplementedError("Cross-attention is not supported.")

        if past_key_values is not None:
            curr_past_key_values = (
                past_key_values.self_attention_cache
                if isinstance(past_key_values, EncoderDecoderCache)
                else past_key_values
            )
        _, B, S, D = hidden_states.shape
        q_hidden, k_hidden, v_hidden = gradient_junction_hook(hidden_states)[1:].view(3, self.num_heads, B, S, D)
        query_states = self._head_wise_projection(q_hidden, self.q_proj)
        key_states = self._head_wise_projection(k_hidden, self.k_proj)
        value_states = self._head_wise_projection(v_hidden, self.v_proj)

        shape = (B, S, -1, self.head_dim)
        query_states = query_states.view(shape).transpose(1, 2)
        key_states = key_states.view(shape).transpose(1, 2)
        value_states = value_states.view(shape).transpose(1, 2)

        if past_key_values is not None:
            key_states, value_states = curr_past_key_values.update(key_states, value_states, self.layer_idx)

        # keep attention interface to unify nnsight self_attn.source tracing
        attention_interface = eager_attention_forward

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attn_dropout.p if self.training else 0.0,
            scaling=self.scaling,
            **kwargs,
        )

        # head-wise attention output
        head_wise_attn_output = torch.einsum('BSHd, DHd -> HBSD', attn_output.detach(), self.out_proj.weight.reshape(-1, self.num_heads, self.head_dim))
        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        attn_output = self.out_proj(attn_output)
        attn_output = torch.cat([attn_output.unsqueeze(0), head_wise_attn_output], dim=0)

        return attn_output, attn_weights
    
    def _head_wise_projection(self, hidden_state: torch.Tensor, proj: nn.Linear) -> torch.Tensor:
        _, B, S, D = hidden_state.shape

        weight_view = proj.weight.view(self.num_heads, self.head_dim, D)
        out = torch.einsum("HBSD, HdD -> BSHd", hidden_state, weight_view)

        if proj.bias is not None:
            out += proj.bias.view(self.num_heads, self.head_dim)
        return out.contiguous()
    

class CustomGPT2Block(GradientCheckpointingLayer):
    def __init__(self, ln_1: nn.Module, attn: nn.Module, ln_2: nn.Module, mlp: nn.Module):
        super().__init__()
        self.ln_1 = ln_1
        self.attn = attn
        self.ln_2 = ln_2
        self.mlp = mlp

    @classmethod
    def from_module(cls, m: nn.Module, **kwargs) -> CustomGPT2Block:
        return cls(ln_1=m.ln_1, attn=m.attn, ln_2=m.ln_2, mlp=m.mlp)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Cache | None = None,
        attention_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        use_cache: bool | None = False,
        **kwargs,
    ) -> torch.Tensor:
        if encoder_hidden_states is not None:
            raise NotImplementedError("Cross-attention is not supported.")

        residual = hidden_states
        hkqv_states = hidden_junction_hook(hidden_states, n=(3* self.attn.num_heads))  # (4, B, N, d_model)
        hkqv_states = self.ln_1(hkqv_states)
        attn_out, _ = self.attn(
            hidden_states=hkqv_states,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = residual + attn_out[0]

        residual = hidden_states
        hgu_states = hidden_junction_hook(hidden_states, n=2)
        hgu_states = self.ln_2(hgu_states)
        mlp_out = self.mlp(hgu_states)
        hidden_states = mlp_out + residual

        return hidden_states
