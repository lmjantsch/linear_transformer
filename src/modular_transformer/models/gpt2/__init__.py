from .gpt2_attention import ModularGPT2Attention
from .gpt2_mlp import ModularGPT2MLP

from ..utils import ArchAccessors

GPT2_ARC = ArchAccessors(
    n_layers=lambda model: model.config.num_hidden_layers,
    n_heads=lambda model: model.config.num_attention_heads,
    model_dim=lambda model: model.config.hidden_size,
    head_dim=lambda model: model.config.hidden_size // model.config.num_attention_heads,

    embed=lambda model: model.transformer.wte,
    pos_emb=lambda model: model.transformer.wpe,
    layers=lambda model: model.transformer.h,
    lm_head=lambda model: model.lm_head,

    pre_ln1=lambda layer: layer.ln_1,
    attn=lambda layer: layer.attn,
    pre_ln2=lambda layer: layer.ln_2,
    mlp=lambda layer: layer.mlp,

    q_proj=lambda layer: layer.attn.q_proj,
    k_proj=lambda layer: layer.attn.k_proj,
    v_proj=lambda layer: layer.attn.v_proj,
    o_proj=lambda layer: layer.attn.out_proj,
    attention_interface=lambda layer: layer.attn.source.attention_interface_0,

    up_proj=lambda layer: layer.mlp.up_proj,
    down_proj=lambda layer: layer.mlp.down_proj,
)