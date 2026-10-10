from typing import Any, Dict
import torch


def validate_dense_tracking_output(output):
    
    for name in ('score_map', 'boxes'):
        values = output[name]
        finite = torch.isfinite(values)
        if not torch.all(finite):
            invalid_index = torch.nonzero(~finite, as_tuple=False)[0].tolist()
            raise ValueError(f'non-finite tracker model output in {name} at index {invalid_index}')


class TrackerOutputPostProcess:
    def start(self):
        pass

    def stop(self):
        pass

    def __call__(self, output: Any) -> Dict[str, Any]:
        raise NotImplementedError()
