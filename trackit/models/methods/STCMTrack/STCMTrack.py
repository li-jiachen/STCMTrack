from typing import Tuple, List, Optional, Mapping, Any
import torch
import torch.nn as nn
from collections import OrderedDict
from timm.layers import trunc_normal_
from trackit.models.backbone.dinov2 import DinoVisionTransformer, interpolate_pos_encoding
from .modules.patch_embed import PatchEmbedNoSizeCheck
from .modules.tmoe.apply import find_all_frozen_nn_linear_names, apply_tmoe
from .modules.head.mlp import MlpAnchorFreeHead
from .modules.ltcp import LocalEnhancedTemporalContextPropagation, LTCPConfig


class STCMTrack_DINOv2(nn.Module):
    def __init__(self, vit: DinoVisionTransformer,
                 template_feat_size: Tuple[int, int],
                 search_region_feat_size: Tuple[int, int],
                 expert_r: int, expert_alpha: float, expert_dropout: float, use_rsexpert: bool = False,
                 expert_nums: int = 4, init_method: str = 'bert', shared_expert: bool = False,
                 route_compression: bool = False, ltcp_config: Optional[dict] = None):
        super().__init__()
        assert template_feat_size[0] <= search_region_feat_size[0] and template_feat_size[1] <= search_region_feat_size[1]
        self.z_size = template_feat_size
        self.x_size = search_region_feat_size

        self.patch_embed = PatchEmbedNoSizeCheck.build(vit.patch_embed)
        self.blocks = vit.blocks
        self.norm = vit.norm
        self.embed_dim = vit.embed_dim

        self.pos_embed = nn.Parameter(torch.empty(1, self.x_size[0] * self.x_size[1], self.embed_dim))
        self.pos_embed.data.copy_(interpolate_pos_encoding(vit.pos_embed.data[:, 1:, :],
                                                           self.x_size,
                                                           vit.patch_embed.patches_resolution,
                                                           num_prefix_tokens=0, interpolate_offset=0))

        self._base_checkpoint_loaded = False
        # TMoE scaling: stored in every checkpoint and compared with the configuration when one is loaded.
        self.register_buffer('_expert_alpha', torch.tensor(float(expert_alpha), dtype=torch.float64))
        self.register_buffer('_use_rsexpert', torch.tensor(bool(use_rsexpert)))
        self.expert_alpha = expert_alpha
        self.use_rsexpert = use_rsexpert

        for name, param in self.named_parameters():
            if not ('.experts.' in name or '.gate' in name):
                param.requires_grad = False
        self.track_query = nn.Parameter(torch.empty(1, self.embed_dim))
        self.query_embed = nn.Parameter(torch.empty(1, self.embed_dim))
        self.token_type_embed = nn.Parameter(torch.empty(3, self.embed_dim))
        trunc_normal_(self.track_query, std=.02)
        trunc_normal_(self.query_embed, std=.02)
        trunc_normal_(self.token_type_embed, std=.02)

        for i_layer, block in enumerate(self.blocks):
            linear_names = find_all_frozen_nn_linear_names(block)
            apply_tmoe(block, linear_names, expert_r, expert_alpha, expert_dropout, use_rsexpert, expert_nums, init_method, shared_expert, route_compression)

        self.head = MlpAnchorFreeHead(self.embed_dim, self.x_size)
        self.ltcp_config = LTCPConfig.from_dict(ltcp_config)
        self.ltcp = LocalEnhancedTemporalContextPropagation(self.embed_dim, self.ltcp_config) if self.ltcp_config.enabled else None
        if self.ltcp_config.enabled and self.ltcp_config.train_only:
            # Training stage 2: only LTCP is optimized; backbone adapters, queries and heads stay frozen.
            self._freeze_except_ltcp()

    def _freeze_except_ltcp(self):
        for name, param in self.named_parameters():
            param.requires_grad = name.startswith('ltcp.')

    def forward(self, z_0: torch.Tensor, z_0_feat_mask: torch.Tensor,
                x_0: torch.Tensor, x_1: torch.Tensor, x_2: torch.Tensor):
        """One first-frame template and three chronological search frames (Sec. 2.1-2.2).

        Each search frame is jointly encoded with the template; the target state token q_t is
        aggregated from the joint features of that frame. LTCP sees 0, 1 and 2 memory frames for
        the three search frames, so the last one has the same two-frame context (m = 2) as
        steady-state inference.
        """
        z_feat = self._z_feat(z_0, z_0_feat_mask)
        memory, outputs = [], []
        for search in (x_0, x_1, x_2):
            raw_tokens, state_token = self._encode_search(z_feat, search)
            tokens = raw_tokens
            if self.ltcp is not None:
                history = torch.stack(memory, dim=1) if memory else None
                tokens = self.ltcp(raw_tokens, history, state_token)
                memory.append(raw_tokens.detach() if self.ltcp_config.detach_memory else raw_tokens)
                memory = memory[-self.ltcp_config.memory_size:]
            outputs.append(self._predict(tokens))  # Eq. (3) goes directly to the heads.
        return tuple(outputs)

    def _predict(self, tokens: torch.Tensor):
        # LTCP cold start passes a strided slice of the joint sequence through, while streaming
        # inference re-assembles a contiguous batch. Both paths feed the heads contiguous tokens so
        # identical values give bit-identical float32 outputs (strided vs contiguous GEMM can differ by 1 ulp).
        return self.head(tokens.contiguous())

    def _z_feat(self, z: torch.Tensor, z_feat_mask: torch.Tensor):
        z = self.patch_embed(z)
        z_W, z_H = self.z_size
        z = z + self.pos_embed.view(1, self.x_size[1], self.x_size[0], self.embed_dim)[:, :z_H, :z_W, :].reshape(1, z_H * z_W, self.embed_dim)
        return z + self.token_type_embed[z_feat_mask.flatten(1)]

    def _x_feat(self, x: torch.Tensor):
        return self.patch_embed(x) + self.pos_embed + self.token_type_embed[2].view(1, 1, self.embed_dim)

    def _encode_search(self, z_feat: torch.Tensor, search: torch.Tensor):
        x_feat = self._x_feat(search)
        query = (self.track_query + self.query_embed).unsqueeze(0).expand(z_feat.size(0), -1, -1)
        joint = torch.cat((query, z_feat, x_feat), dim=1)
        for block in self.blocks:
            joint = block(joint)
        joint = self.norm(joint)
        return joint[:, -x_feat.size(1):], joint[:, :1]

    def export_ltcp_state_dict(self):
        """Explicit adapter export; state_dict() always contains the full model."""
        if self.ltcp is None:
            raise ValueError('Cannot export LTCP from a base-only model')
        return OrderedDict((key, value) for key, value in self.state_dict().items()
                           if key.startswith('ltcp.') or key in self._checkpoint_metadata_keys)

    _checkpoint_metadata_keys = ('_expert_alpha', '_use_rsexpert')

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, **kwargs):
        # Top-level entries whose names start with an underscore are checkpoint metadata, not network weights;
        # the TMoE scaling is the only metadata this model reads.
        state = OrderedDict((key, value) for key, value in state_dict.items()
                            if not key.startswith('_') or key in self._checkpoint_metadata_keys)
        # Everything is validated before the first tensor is copied.
        if any(key not in state for key in self._checkpoint_metadata_keys):
            raise ValueError('Not an STCMTrack checkpoint: the TMoE scaling entries (_expert_alpha, _use_rsexpert) '
                             'are missing. Use a checkpoint written by train_stcmtrack.sh or '
                             'tools/export_stcmtrack_weights.py.')
        for key in self._checkpoint_metadata_keys:
            if state[key].item() != getattr(self, key).item():
                raise ValueError(f'Checkpoint/configuration mismatch: {key}')
        supplied_ltcp = {key for key in state if key.startswith('ltcp.')}
        expected_ltcp = {key for key in self.state_dict() if key.startswith('ltcp.')}
        if supplied_ltcp and supplied_ltcp != expected_ltcp:
            raise ValueError('Incomplete or incompatible LTCP gate in checkpoint')
        payload = set(state) - set(self._checkpoint_metadata_keys)
        adapter_only = bool(payload) and all(key.startswith('ltcp.') for key in payload)
        if adapter_only:
            if not self._base_checkpoint_loaded:
                raise ValueError('Load the matching full base checkpoint before the LTCP adapter')
            if self.ltcp is None or payload != {key for key in self.state_dict() if key.startswith('ltcp.')}:
                raise ValueError('LTCP adapter checkpoint does not match this model')
        else:
            required = {key for key in self.state_dict() if not key.startswith('ltcp.')}
            missing = required - set(state)
            if missing:
                raise ValueError(f'Incomplete full-model checkpoint: {sorted(missing)[:8]}')
        result = super().load_state_dict(state, strict=strict, **kwargs)
        if any(key.startswith('head.') for key in state):
            self._base_checkpoint_loaded = True
        return result
