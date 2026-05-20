from __future__ import annotations

import torch
import torch.nn as nn

from linear_transformer.modules import ACT_FN, BILINEAR_FN

class ModularGemma2MLP(nn.Module):

    def __init__(self, gate_proj: nn.Linear, up_proj: nn.Linear, down_proj: nn.Linear, act_fn: callable, mul_fn: callable) -> None:
        super().__init__()
        self.gate_proj = gate_proj   # (d_ffn, d_model)
        self.up_proj = up_proj       # (d_ffn, d_model)
        self.down_proj = down_proj   # (d_model, d_ffn)
        self.act_fn = act_fn
        self.mul_fn = mul_fn

    @classmethod
    def from_module(cls, m: nn.Module, **kwargs: dict | None) -> ModularGemma2MLP:
        act_fn = ACT_FN[kwargs.get('mlp_act_fn', 'gelu_tanh')]
        mul_fn = BILINEAR_FN[kwargs.get('mul_fn', 'mul')]
        return cls(
            gate_proj=m.gate_proj,
            up_proj=m.up_proj,
            down_proj=m.down_proj,
            act_fn=act_fn,
            mul_fn=mul_fn,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_out = self.gate_proj(x)
        gate_act = self.act_fn(gate_out)
        up_out = self.up_proj(x)

        z = self.mul_fn(gate_act, up_out)
        output = self.down_proj(z)
        return output
