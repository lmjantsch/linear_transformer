from .gemma2_attention import ModularGemma2Attention
from .gemma2_mlp import ModularGemma2MLP
from .gemma2_rmsnorm import ModularGemma2RMSNorm

from ..base import ArchAccessors


GEMMA2_ARC = ArchAccessors(
    n_layers=lambda model: model.config.num_hidden_layers,
    n_heads=lambda model: model.config.num_attention_heads,
    model_dim=lambda model: model.config.hidden_size,
    head_dim=lambda model: model.config.head_dim,
    uses_rotary_emb = lambda model: True,

    embed=lambda model: model.model.embed_tokens,
    rotary_emb=lambda model: model.model.rotary_emb,
    layers=lambda model: model.model.layers,
    ln=lambda model: model.model.norm,
    lm_head=lambda model: model.lm_head,

    pre_ln1=lambda layer: layer.input_layernorm,
    attn=lambda layer: layer.self_attn,
    post_ln1=lambda layer: layer.post_attention_layernorm,
    pre_ln2=lambda layer: layer.pre_feedforward_layernorm,
    mlp=lambda layer: layer.mlp,
    post_ln2=lambda layer: layer.post_feedforward_layernorm,

    q_proj=lambda layer: layer.self_attn.q_proj,
    k_proj=lambda layer: layer.self_attn.k_proj,
    v_proj=lambda layer: layer.self_attn.v_proj,
    o_proj=lambda layer: layer.self_attn.o_proj,
    attention_interface=lambda layer: layer.self_attn.source.self_attention_interface_0,

    up_proj=lambda layer: layer.mlp.up_proj,
    gate_proj=lambda layer: layer.mlp.gate_proj,
    down_proj=lambda layer: layer.mlp.down_proj,
)