from abc import ABC, abstractmethod
from typing import Iterator, get_type_hints

import torch
from torch import nn

from linear_transformer.adapter.accessors import ArchAccessors, MODEL_SCOPE, LAYER_SCOPE

class Adapter(ABC):

    def _attach_arch_accessor(self, arch, module, scope):
        hints = get_type_hints(arch, include_extras=True)
        
        for name, hint in hints.items():
            func = getattr(arch, name)
            metadata = getattr(hint, "__metadata__", ())
            
            if scope in metadata:
                setattr(self, name, func(module))

class ModelAdapter(Adapter):

    def __init__(self, model, arch: ArchAccessors):
        self.model = model
        self.config = model.config
        self.device = model.device
        self.dtype = model.dtype

        self.arch = arch

        self._attach_arch_accessor(arch, model, MODEL_SCOPE)

    @property
    def source_dims(self) -> int:
        return (1 + self.n_layers * (self.n_heads + 1))

    @property
    def grad_dims(self) -> int:
        return (self.n_layers * (3 * self.n_heads + 1) + 1)
    
    @property
    def wrapped_layers(self) -> Iterator['LayerAdapter']:
        for idx, layer in enumerate(self.layers):
            yield LayerAdapter(
                layer=layer, 
                layer_idx=idx, 
                arch=self.arch,
                model_reference=self
            )
    
    # TODO string types is not clean
    def get_src_slice(self, type: str, layer_id: int | None) -> slice:
        if type == 'emb':
            return slice(0, 1)
        if type.startswith('attn'):
            start = 1 + layer_id * (self.n_heads + 1)
            return slice(start, start + self.n_heads)
        if type == 'mlp':
            start = 1 + layer_id * (self.n_heads + 1) + self.n_heads
            return slice(start, start + 1)
        if type == 'lm_head': # needs to be included for '_update_score' lookup
            start = 1 + self.n_layers * (self.n_heads + 1)  # TODO Wrong? + self.n_heads
            return slice(start, None)
        raise NotImplementedError

    def get_tgt_slice(self, type: str, layer_id: int | None) -> slice:
        if type == 'lm_head':
            return slice(-1, None)
        elif type == 'mlp':
            start = layer_id * (3 * self.n_heads + 1) + 3 * self.n_heads
            return slice(start, start + 1)
        elif type == 'attn_v':
            start = layer_id * (3 * self.n_heads + 1) + 2 * self.n_heads
            return slice(start, start + self.n_heads)
        elif type == 'attn_k':
            start = layer_id * (3 * self.n_heads + 1) + self.n_heads
            return slice(start, start + self.n_heads)
        elif type == 'attn_q':
            start = layer_id * (3 * self.n_heads + 1)
            return slice(start, start + self.n_heads)
        else:
            raise NotImplementedError(f"Type '{type}' is not implemented.")

    def emb_output(self, detached=True):
        # Embedding source is input of first layer (GPT2 has 2 different embeddings)
        first_layer = next(self.wrapped_layers)
        return first_layer.residual_pre_state(detached=detached)
    
    def logits_input(self, detached=False):
        if detached:
            return self.lm_head.input.detach()
        return self.lm_head.input
    
    def logits(self, detached=False):
        if detached:
            return self.lm_head.output.detach()
        return self.lm_head.output
        

class LayerAdapter(Adapter):

    def __init__(self, layer, layer_idx: int, model_reference: ModelAdapter, arch: ArchAccessors):
        self.layer = layer
        self.layer_idx = layer_idx
        self.model_reference = model_reference

        self._attach_arch_accessor(arch, layer, LAYER_SCOPE)

    def residual_pre_state(self, detached=True) -> torch.Tensor:
        if detached:
            return self.layer.source.hidden_junction_hook_0.input.detach()
        return self.layer.source.hidden_junction_hook_0.input
    
    def residual_mid_state(self, detached=True) -> torch.Tensor:
        if detached:
            return self.layer.source.hidden_junction_hook_1.input.detach()
        return self.layer.source.hidden_junction_hook_1.input
    
    def residual_post_state(self, detached=True) -> torch.Tensor:
        if detached:
            return self.layer.output.detach()
        return self.layer.output 
    
    def attn_output(self, detached=True) -> torch.Tensor:
        if detached:
            return self.attn.output[0][0].detach()
        return self.attn.output[0][0]
    
    def head_wise_attn_output(self, detached=True) -> torch.Tensor:
        if detached:
            return self.attn.output[0][1:].detach()
        return self.attn.output[0][1:]
    
    def mlp_output(self, detached=True) -> torch.Tensor:
        if detached:
            return self.mlp.output.detach()
        return self.mlp.output
    
    def attn_input(self, detached=False) -> torch.Tensor: # B, S, D
        if detached:
            return self.layer.source.hidden_junction_hook_0.output[0].detach()
        return self.layer.source.hidden_junction_hook_0.output[0]

    def qkv_input(self, detached=False) -> torch.Tensor: # 3, B, S, D
        if detached:
            return self.layer.source.hidden_junction_hook_0.output[1:].detach()
        return self.layer.source.hidden_junction_hook_0.output[1:]
    
    def mlp_input(self, detached=False) -> torch.Tensor: # B, S, D
        if detached:
            return self.layer.source.hidden_junction_hook_1.output[0].detach()
        return self.layer.source.hidden_junction_hook_1.output[0]
    
    def gu_input(self, detached=False) -> torch.Tensor: # 2, B, S, D
        if detached:
            return self.layer.source.hidden_junction_hook_1.output[1:].detach()
        return self.layer.source.hidden_junction_hook_1.output[1:]