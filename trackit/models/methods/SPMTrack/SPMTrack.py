"""SPMTrack baseline: training / shared model.

Ported from the official SPMTrack implementation, pinned to
    https://github.com/WenRuiCai/SPMTrack  commit c581fe27231f3e16c38578e47daddadfaf6ffd7d
    (trackit/models/methods/SPMTrack/SPMTrack.py), Apache-2.0.

This is the independent baseline path for the comparison against STCMTrack. It does not
import anything from trackit.models.methods.STCMTrack, and it contains no LTCP, MCC or RGTC.
The network structure follows the pinned commit: three templates (first frame plus two
history references), two search frames, a query state that is propagated from the first
search frame to the second, and a head input re-weighted by the query state.

Deviations from the pinned commit (engineering only; the forward computation is unchanged):
  1. `track_query` and `query_embed` are initialized explicitly with a truncated normal
     (std 0.02) like `token_type_embed`. Upstream leaves them as `torch.empty`, i.e.
     undefined memory, with a code comment that training sometimes does not converge.
  2. `load_state_dict` validates a checkpoint before copying any tensor: STCMTrack
     checkpoints are rejected, every trainable parameter must be present (nothing is filled
     randomly), unknown keys are rejected, and a TMoE alpha / rs-expert mismatch is an error
     (upstream silently ignores the alpha stored in the checkpoint).
  3. Provenance: checkpoints written by this class carry `_spmtrack_port_version`. A
     checkpoint without it (official SPMTrack release or anything else) is only accepted when
     the config sets `model.allow_unmarked_weights: true`. No marker is ever added to an
     existing file.
  4. The unused upstream `_fusion` helper is omitted.
"""
import math
import warnings
from collections import OrderedDict
from typing import Any, Mapping, Tuple

import torch
import torch.nn as nn
from timm.layers import trunc_normal_

from trackit.models.backbone.dinov2 import DinoVisionTransformer, interpolate_pos_encoding
from .modules.patch_embed import PatchEmbedNoSizeCheck
from .modules.tmoe.apply import find_all_frozen_nn_linear_names, apply_tmoe
from .modules.head.mlp import MlpAnchorFreeHead

UPSTREAM_REPOSITORY = 'https://github.com/WenRuiCai/SPMTrack'
UPSTREAM_COMMIT = 'c581fe27231f3e16c38578e47daddadfaf6ffd7d'
SPMTRACK_PORT_VERSION = 1
PORT_VERSION_KEY = '_spmtrack_port_version'

_STCMTRACK_ONLY_KEYS = ('_expert_alpha', '_use_rsexpert')


class SPMTrack_DINOv2(nn.Module):
    def __init__(self, vit: DinoVisionTransformer,
                 template_feat_size: Tuple[int, int],
                 search_region_feat_size: Tuple[int, int],
                 expert_r: int, expert_alpha: float, expert_dropout: float, use_rsexpert: bool = False,
                 expert_nums: int = 4, init_method: str = 'bert', shared_expert: bool = False, route_compression: bool = False,
                 allow_unmarked_weights: bool = False):
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

        self.expert_alpha = expert_alpha
        self.use_rsexpert = use_rsexpert
        self.allow_unmarked_weights = allow_unmarked_weights

        for name, param in self.named_parameters():
            if not ('.experts.' in name or '.gate' in name):
                param.requires_grad = False
        self.track_query = nn.Parameter(torch.empty(1, self.embed_dim))
        self.query_embed = nn.Parameter(torch.empty(1, self.embed_dim))
        self.token_type_embed = nn.Parameter(torch.empty(3, self.embed_dim))
        trunc_normal_(self.token_type_embed, std=.02)
        
        trunc_normal_(self.track_query, std=.02)
        trunc_normal_(self.query_embed, std=.02)

        for i_layer, block in enumerate(self.blocks):
            linear_names = find_all_frozen_nn_linear_names(block)
            apply_tmoe(block, linear_names, expert_r, expert_alpha, expert_dropout, use_rsexpert, expert_nums, init_method, shared_expert, route_compression)

        self.head = MlpAnchorFreeHead(self.embed_dim, self.x_size)

        
        
        self._trainable_parameter_names = frozenset(name for name, param in self.named_parameters() if param.requires_grad)

    def forward(self, z_0: torch.Tensor, z_1: torch.Tensor, z_2: torch.Tensor,
                x_0: torch.Tensor, x_1: torch.Tensor,
                z_0_feat_mask: torch.Tensor, z_1_feat_mask: torch.Tensor, z_2_feat_mask: torch.Tensor):
        z0_feat = self._z_feat(z_0, z_0_feat_mask)
        z1_feat = self._z_feat(z_1, z_1_feat_mask)
        z2_feat = self._z_feat(z_2, z_2_feat_mask)
        x0_feat = self._x_feat(x_0)
        x1_feat = self._x_feat(x_1)
        return self._multi_frame_predict(z0_feat, z1_feat, z2_feat, x0_feat, x1_feat)

    def _z_feat(self, z: torch.Tensor, z_feat_mask: torch.Tensor):
        z = self.patch_embed(z)
        z_W, z_H = self.z_size
        z = z + self.pos_embed.view(1, self.x_size[1], self.x_size[0], self.embed_dim)[:, : z_H, : z_W, :].reshape(1, z_H * z_W, self.embed_dim)
        z = z + self.token_type_embed[z_feat_mask.flatten(1)]
        return z

    def _x_feat(self, x: torch.Tensor):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = x + self.token_type_embed[2].view(1, 1, self.embed_dim)
        return x

    def _multi_frame_predict(self, z_0, z_1, z_2, x_0, x_1):
        z_feat = torch.cat([z_0, z_1, z_2], dim=1)
        B, N, D = z_0.shape

        new_query = self.track_query.unsqueeze(0).expand(B, 1, -1)
        query = new_query + self.query_embed.unsqueeze(0)
        fusion_feat = torch.cat((query, z_feat, x_0), dim=1)

        for i in range(len(self.blocks)):
            fusion_feat = self.blocks[i](fusion_feat)
        fusion_feat = self.norm(fusion_feat)
        enc_opt = fusion_feat[:, -x_0.size(1):, ...]

        track_query = fusion_feat[:, :1, ...].clone().detach()
        att = torch.matmul(enc_opt, fusion_feat[:, :1, ...].transpose(1, 2))
        opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 1, 2)).contiguous().squeeze(1)
        output1 = self.head(opt)

        query_2 = track_query + new_query + self.query_embed.unsqueeze(0)
        fusion_feat2 = torch.cat((query_2, z_feat, x_1), dim=1)

        for i in range(len(self.blocks)):
            fusion_feat2 = self.blocks[i](fusion_feat2)
        fusion_feat2 = self.norm(fusion_feat2)
        enc_opt2 = fusion_feat2[:, -x_1.size(1):, ...]

        att2 = torch.matmul(enc_opt2, fusion_feat2[:, :1, ...].transpose(1, 2))
        opt2 = (enc_opt2.unsqueeze(-1) * att2.unsqueeze(-2)).permute((0, 3, 1, 2)).contiguous().squeeze(1)
        output2 = self.head(opt2)

        return output1, output2

    
    def state_dict(self, *, destination=None, prefix='', keep_vars=False):
        
        state_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        for name, _ in self.named_parameters():
            if name not in self._trainable_parameter_names:
                state_dict.pop(prefix + name, None)
        state_dict[prefix + 'expert_alpha'] = torch.as_tensor(self.expert_alpha)
        state_dict[prefix + 'use_rsexpert'] = torch.as_tensor(self.use_rsexpert)
        state_dict[prefix + PORT_VERSION_KEY] = torch.as_tensor(SPMTRACK_PORT_VERSION, dtype=torch.int64)
        return state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        
        
        state_dict = OrderedDict(state_dict)
        self._check_checkpoint_provenance(state_dict)
        expert_alpha = state_dict.pop('expert_alpha', None)
        use_rsexpert = state_dict.pop('use_rsexpert', None)
        state_dict.pop(PORT_VERSION_KEY, None)
        if expert_alpha is not None and not math.isclose(float(expert_alpha), float(self.expert_alpha), rel_tol=1e-6):
            raise ValueError(f'Checkpoint TMoE alpha {float(expert_alpha)} differs from the configured {self.expert_alpha}; '
                             f'the experts were trained with a different scaling.')
        if use_rsexpert is not None and bool(use_rsexpert) != bool(self.use_rsexpert):
            raise ValueError('Checkpoint use_rsexpert differs from the configured TMoE scaling mode')
        known_keys = set(super().state_dict().keys())
        unexpected = sorted(key for key in state_dict if key not in known_keys)
        if unexpected:
            raise ValueError(f'Unexpected keys for SPMTrack: {unexpected[:8]}{" ..." if len(unexpected) > 8 else ""}')
        missing = sorted(self._trainable_parameter_names - set(state_dict))
        if missing:
            raise ValueError(f'The checkpoint lacks trainable SPMTrack parameters (they are never filled randomly): '
                             f'{missing[:8]}{" ..." if len(missing) > 8 else ""}')
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _check_checkpoint_provenance(self, state_dict: Mapping[str, Any]):
        foreign = sorted(key for key in state_dict if key in _STCMTRACK_ONLY_KEYS or key.startswith('ltcp.'))
        if foreign:
            raise ValueError(
                f'This is an STCMTrack checkpoint (found {foreign[:4]}). STCMTrack is a single-template model '
                f'with different state propagation and prediction path; its weights are not SPMTrack weights.')
        if PORT_VERSION_KEY in state_dict:
            version = int(state_dict[PORT_VERSION_KEY])
            if version != SPMTRACK_PORT_VERSION:
                raise ValueError(f'Unsupported {PORT_VERSION_KEY}={version}; this code supports {SPMTRACK_PORT_VERSION}')
            return
        if not self.allow_unmarked_weights:
            raise ValueError(
                f'The checkpoint has no {PORT_VERSION_KEY} marker. Set model.allow_unmarked_weights: true to load '
                f'a checkpoint that was trained with the official SPMTrack code '
                f'({UPSTREAM_REPOSITORY} @ {UPSTREAM_COMMIT[:7]}).')
        warnings.warn(f'Loading a checkpoint without the {PORT_VERSION_KEY} marker as official SPMTrack weights '
                      f'(model.allow_unmarked_weights).', stacklevel=3)
