"""SPMTrack baseline: streaming inference model.

Corresponds to `SPMTrackBaseline_DINOv2` in the official
trackit/models/methods/SPMTrack/SPMTrack_full_finetune.py @ c581fe27231f3e16c38578e47daddadfaf6ffd7d
(WenRuiCai/SPMTrack, Apache-2.0): `forward_tracking` with a per-sequence query state.

The query used for a frame is  state(previous frame) + track_query + query_embed,  and for the
first tracked frame the state is zero. This matches the second search frame of the training
forward (`query_2 = track_query + new_query + query_embed`). The new state is the detached
query output of the joint encoder and is stored after every frame.

Deviations from the pinned commit (engineering only):
  * This class inherits the training model, so TMoE is built in the constructor. Upstream
    applies TMoE inside `load_state_dict`, which wraps the layers a second time whenever the
    cached inference model is refreshed from the training model.
  * The three template inputs and masks are required. Upstream silently substitutes copies of
    earlier templates when they are missing (and its source carries a "has a bug" comment).
  * The query state is a plain dictionary created in `init_eval`, cleared per sequence by
    `reset_tracking` / `forget_tracking` (called by the evaluator), and released in `end_eval`.
    Upstream pre-fills `range(total_sequence_num)` and never drops finished sequences.
  * `dinov2_full_finetune` is not supported: the upstream builder cannot construct it (it
    omits the TMoE constructor arguments of that class).
"""
from typing import Dict, List, Optional

import torch

from .SPMTrack import SPMTrack_DINOv2


class SPMTrackInference_DINOv2(SPMTrack_DINOv2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.track_query_dicts: Dict[int, torch.Tensor] = {}

    def init_eval(self, total_sequence_num):
        self.track_query_dicts = {}

    def end_eval(self):
        self.track_query_dicts = {}

    def reset_tracking(self, task_id):
        """A (re-)initialized sequence starts with a zero query state."""
        self.track_query_dicts.pop(task_id, None)

    def forget_tracking(self, task_id):
        self.track_query_dicts.pop(task_id, None)

    def forward_tracking(self, ids: List[int], z_0: torch.Tensor, x: torch.Tensor, z_0_feat_mask: torch.Tensor,
                         z_1: torch.Tensor, z_2: torch.Tensor,
                         z_1_feat_mask: torch.Tensor, z_2_feat_mask: torch.Tensor):
        B = len(ids)
        if B != x.size(0) or len(set(ids)) != B:
            raise ValueError('Each batch row must have a unique sequence id')
        z_feat = torch.cat([self._z_feat(z_0, z_0_feat_mask),
                            self._z_feat(z_1, z_1_feat_mask),
                            self._z_feat(z_2, z_2_feat_mask)], dim=1)
        x = self._x_feat(x)

        new_query = []
        for task_id in ids:
            state: Optional[torch.Tensor] = self.track_query_dicts.get(task_id)
            if state is not None:
                new_query.append(state.unsqueeze(0))
            else:
                new_query.append(torch.zeros_like(self.track_query).to(z_feat.device).unsqueeze(0))
        new_query = torch.cat(new_query, dim=0)
        query = new_query + self.track_query.unsqueeze(0).expand(B, 1, -1) + self.query_embed.unsqueeze(0)

        fusion_feat = torch.cat((query, z_feat, x), dim=1)
        for i in range(len(self.blocks)):
            fusion_feat = self.blocks[i](fusion_feat)
        fusion_feat = self.norm(fusion_feat)

        enc_opt = fusion_feat[:, -x.size(1):, ...]
        track_query = fusion_feat[:, :1, ...].clone().detach()

        for i, task_id in enumerate(ids):
            self.track_query_dicts[task_id] = track_query[i].clone()

        att = torch.matmul(enc_opt, fusion_feat[:, :1, ...].transpose(1, 2))
        opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 1, 2)).contiguous().squeeze(1)
        return self.head(opt)
