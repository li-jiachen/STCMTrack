from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TemporalTokenContextPropagationConfig:
    enabled: bool = False
    train_only: bool = False
    memory_size: int = 15
    detach_memory: bool = True
    eps: float = 1e-6
    frame_softmax_temperature: float = 1.0
    gate_bias_init: float = -4.0
    confidence_weight_init: float = 2.0
    max_gate: float = 0.05
    store_enhanced_memory: bool = False
    memory_device: str = "cpu"
    memory_dtype: str = "float16"
    print_summary: bool = True

    @classmethod
    def from_dict(cls, config: Optional[dict]):
        if config is None:
            return cls()
        valid_names = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in config.items() if key in valid_names})


@dataclass
class _LTCPStats:
    calls: int = 0
    memory_calls: int = 0
    memory_frames_seen: int = 0
    stored_tokens: int = 0


class TemporalTokenContextPropagation(nn.Module):
    def __init__(self, embed_dim: int, config: TemporalTokenContextPropagationConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(embed_dim + 1, 1)
        self._stats = _LTCPStats()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, self.config.gate_bias_init)
        with torch.no_grad():
            self.gate.weight[:, 0].fill_(self.config.confidence_weight_init)

    def reset_statistics(self):
        self._stats = _LTCPStats()

    def observe_stored_tokens(self, count: int):
        self._stats.stored_tokens += int(count)

    def format_summary(self):
        if not self.config.print_summary:
            return []
        stats = self._stats
        avg_memory_frames = stats.memory_frames_seen / stats.memory_calls if stats.memory_calls > 0 else 0.0
        return [
            'LTCP summary:',
            f'  calls={stats.calls} memory_calls={stats.memory_calls} '
            f'avg_memory_frames={avg_memory_frames:.2f} memory_size={self.config.memory_size}',
            f'  detach_memory={self.config.detach_memory} store_enhanced_memory={self.config.store_enhanced_memory} '
            f'memory_device={self.config.memory_device} memory_dtype={self.config.memory_dtype}',
            f'  stored_token_snapshots={stats.stored_tokens}',
        ]

    def forward(self, x: torch.Tensor, memory: Optional[torch.Tensor], state_token: torch.Tensor):
        self._stats.calls += 1
        max_gate = max(float(self.config.max_gate), 0.0)
        if max_gate == 0.0:
            return x
        if self.config.memory_size <= 0:
            return x
        if memory is None or memory.numel() == 0:
            return x
        if memory.ndim == 3:
            memory = memory.unsqueeze(1)
        if memory.ndim != 4:
            raise ValueError(f'LTCP memory must be [B, T, N, D] or [B, N, D], got {tuple(memory.shape)}')
        if memory.size(1) == 0:
            return x
        if memory.size(0) != x.size(0) or memory.size(2) != x.size(1) or memory.size(3) != x.size(2):
            raise ValueError(f'LTCP shape mismatch: x={tuple(x.shape)} memory={tuple(memory.shape)}')
        if state_token.ndim != 3 or state_token.size(0) != x.size(0) or state_token.size(1) != 1 or state_token.size(2) != x.size(2):
            raise ValueError(f'LTCP state_token must be [B, 1, D], got {tuple(state_token.shape)} for x={tuple(x.shape)}')

        memory = memory[:, -self.config.memory_size:, :, :]
        if self.config.detach_memory:
            memory = memory.detach()
        memory = torch.nan_to_num(memory, nan=0.0, posinf=0.0, neginf=0.0)
        self._stats.memory_calls += 1
        self._stats.memory_frames_seen += int(memory.size(1))

        x_norm = F.normalize(x.float(), p=2, dim=-1, eps=self.config.eps)
        memory_norm = F.normalize(memory.float(), p=2, dim=-1, eps=self.config.eps)

        pixel_similarity = (x_norm.unsqueeze(1) * memory_norm).sum(dim=-1).amax(dim=1)
        local_conf = ((pixel_similarity + 1.0) * 0.5).clamp_(0.0, 1.0)

        x_frame = F.normalize(x.float().mean(dim=1), p=2, dim=-1, eps=self.config.eps)
        memory_frame = F.normalize(memory.float().mean(dim=2), p=2, dim=-1, eps=self.config.eps)
        frame_similarity = (memory_frame * x_frame.unsqueeze(1)).sum(dim=-1)
        temperature = max(float(self.config.frame_softmax_temperature), self.config.eps)
        frame_weight = torch.softmax(frame_similarity / temperature, dim=1)

        memory_tokens = torch.einsum('bt,btnd->bnd', frame_weight.to(memory.dtype), memory)
        frame_similarity_agg = (frame_weight * frame_similarity).sum(dim=1, keepdim=True)
        global_conf = ((frame_similarity_agg + 1.0) * 0.5).clamp_(0.0, 1.0)

        temporal_confidence = (local_conf * global_conf).unsqueeze(-1).to(dtype=state_token.dtype)
        state_token_expand = state_token.expand(-1, x.size(1), -1)
        gate_input = torch.cat((temporal_confidence, state_token_expand), dim=-1)
        gate = torch.sigmoid(self.gate(gate_input)).to(dtype=x.dtype)
        gate = gate * max_gate
        return x * (1.0 - gate) + memory_tokens.to(dtype=x.dtype) * gate


def get_ltcp_memory_dtype(dtype_name: str, fallback: torch.dtype):
    if dtype_name is None or dtype_name == "same":
        return fallback
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f'Unsupported LTCP memory dtype: {dtype_name}')
