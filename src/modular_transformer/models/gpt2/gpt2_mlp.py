from __future__ import annotations

import torch
import torch.nn as nn

from modular_transformer.modules import ACT_FN
from modular_transformer.models.utils import conv1d_to_linear


class ModularGPT2MLP(nn.Module):

    def __init__(self, up_proj: nn.Linear, down_proj: nn.Linear, act_fn: callable) -> None:
        super().__init__()
        self.up_proj = up_proj      # (d_ffn, d_model) as nn.Linear
        self.down_proj = down_proj  # (d_model, d_ffn) as nn.Linear
        self.act_fn = act_fn

    @classmethod
    def from_module(cls, m: nn.Module, **kwargs) -> ModularGPT2MLP:
        return cls(
            up_proj=conv1d_to_linear(m.c_fc),
            down_proj=conv1d_to_linear(m.c_proj),
            act_fn=ACT_FN[kwargs.get('mlp_act_fn', 'gelu_tanh')],
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.up_proj(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.down_proj(hidden_states)
        return hidden_states
