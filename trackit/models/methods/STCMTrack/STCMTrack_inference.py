
from typing import List
import torch
from .STCMTrack import STCMTrack_DINOv2
from .modules.ltcp import get_ltcp_memory_dtype


class STCMTrackInference_DINOv2(STCMTrack_DINOv2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ltcp_memory_dicts = {}

    def init_eval(self, total_sequence_num=None):
        self.ltcp_memory_dicts.clear()
        if self.ltcp is not None:
            self.ltcp.reset_statistics()

    def reset_tracking(self, task_id):
        self.ltcp_memory_dicts.pop(task_id, None)

    def forget_tracking(self, task_id):
        self.ltcp_memory_dicts.pop(task_id, None)

    def end_eval(self):
        if self.ltcp is not None:
            for line in self.ltcp.format_summary():
                print(line, flush=True)
        self.ltcp_memory_dicts.clear()

    def forward_tracking(self, ids: List[int], z_0: torch.Tensor, x: torch.Tensor,
                         z_0_feat_mask: torch.Tensor):
        if len(ids) != x.size(0) or len(set(ids)) != len(ids):
            raise ValueError('Each batch row must have a unique sequence id')
        raw_tokens, state_token = self._encode_search(self._z_feat(z_0, z_0_feat_mask), x)
        tokens = raw_tokens
        if self.ltcp is not None:
            tokens = self._apply_ltcp_tracking(ids, raw_tokens, state_token)
        return self._predict(tokens)

    def _apply_ltcp_tracking(self, ids, raw_tokens, state_token):
        outputs = []
        for row, task_id in enumerate(ids):
            snapshots = self.ltcp_memory_dicts.get(task_id, [])
            history = (torch.stack(snapshots).unsqueeze(0).to(raw_tokens)
                       if snapshots else None)
            outputs.append(self.ltcp(raw_tokens[row:row + 1], history, state_token[row:row + 1]))
        
        dtype = get_ltcp_memory_dtype(self.ltcp_config.memory_dtype, raw_tokens.dtype)
        device = 'cpu' if self.ltcp_config.memory_device == 'cpu' else raw_tokens.device
        for row, task_id in enumerate(ids):
            snapshots = self.ltcp_memory_dicts.setdefault(task_id, [])
            snapshots.append(raw_tokens[row].detach().to(device=device, dtype=dtype).clone())
            del snapshots[:-self.ltcp_config.memory_size]
            self.ltcp.observe_stored_tokens(1)
        return torch.cat(outputs, dim=0)
