from dataclasses import dataclass
from typing import Callable, Annotated, Any

MODEL_SCOPE = "model"
LAYER_SCOPE = "layer"

ModelCallable = Annotated[Callable[[Any], Any], MODEL_SCOPE]
LayerCallable = Annotated[Callable[[Any], Any], LAYER_SCOPE]


class ArchAccessError(Exception):
    """Raised when module does not exist on architecture"""
    def __init__(self, module_name: str):
        custom_message = f"Module '{module_name}' does not exist on this model"
        super().__init__(custom_message)

def raise_arch_access_error(module_name:str):
    raise ArchAccessError(module_name)


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


_LLAMA_LIKE = ArchAccessors(
    n_layers=lambda model: model.config.num_hidden_layers,
    n_heads=lambda model: model.config.num_attention_heads,
    model_dim=lambda model: model.config.hidden_size,
    head_dim=lambda model: getattr(model.config, 'head_dim', model.config.hidden_size // model.config.num_attention_heads),
    uses_rotary_emb = lambda model: True,

    embed=lambda model: model.model.embed_tokens,
    rotary_emb=lambda model: model.model.rotary_emb,
    layers=lambda model: model.model.layers,
    lm_head=lambda model: model.lm_head,

    pre_ln1=lambda layer: layer.input_layernorm,
    attn=lambda layer: layer.self_attn,
    pre_ln2=lambda layer: layer.post_attention_layernorm,
    mlp=lambda layer: layer.mlp,

    q_proj=lambda layer: layer.self_attn.q_proj,
    k_proj=lambda layer: layer.self_attn.k_proj,
    v_proj=lambda layer: layer.self_attn.v_proj,
    o_proj=lambda layer: layer.self_attn.o_proj,
    attention_interface=lambda layer: layer.self_attn.source.attention_interface_0,

    up_proj=lambda layer: layer.mlp.up_proj,
    gate_proj=lambda layer: layer.mlp.gate_proj,
    down_proj=lambda layer: layer.mlp.down_proj,
)

_GPT2 = ArchAccessors(
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

_GEMMA2 = ArchAccessors(
    n_layers=lambda model: model.config.num_hidden_layers,
    n_heads=lambda model: model.config.num_attention_heads,
    model_dim=lambda model: model.config.hidden_size,
    head_dim=lambda model: model.config.head_dim,
    uses_rotary_emb = lambda model: True,

    embed=lambda model: model.model.embed_tokens,
    rotary_emb=lambda model: model.model.rotary_emb,
    layers=lambda model: model.model.layers,
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
    attention_interface=lambda layer: layer.self_attn.source.attention_interface_0,

    up_proj=lambda layer: layer.mlp.up_proj,
    gate_proj=lambda layer: layer.mlp.gate_proj,
    down_proj=lambda layer: layer.mlp.down_proj,
)

def get_arch(model_id: str) -> ArchAccessors:
    """Map model ID to architecture accessors."""
    if model_id in ['gpt2']:
        return _GPT2
    if model_id in ['Qwen/Qwen2.5-0.5B', 'meta-llama/Llama-3.1-8B']:
        return _LLAMA_LIKE
    if model_id in ['google/gemma-2-2b']:
        return _GEMMA2
    raise NotImplementedError(f"Model '{model_id}' has no assigned ArchAccessor")