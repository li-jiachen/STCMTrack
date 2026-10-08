"""Foreground-indicating template masks for the three SPMTrack template slots.

Ported from the official
.../default/pipelines/_common/template_foreground_indicating_mask_generation.py
@ WenRuiCai/SPMTrack c581fe27231f3e16c38578e47daddadfaf6ffd7d (Apache-2.0).

It must be listed after `SPMTrackOneStream_Evaluation_MainPipeline`: it reads the reference
selection and the new online templates that the main pipeline puts into
`context.temporary_objects`, so every mask belongs to exactly the frame whose template it describes.

Deviations (engineering only): the device is a constructor argument instead of the literal `.cuda()`;
the reference selection is not recomputed here (see the main pipeline); the mask memory is reset on
(re-)initialization and released when the sequence ends (upstream never releases it).
"""
from typing import Tuple, Dict, Any, List
import numpy as np
import torch

from trackit.core.utils.bbox_mask_gen import get_foreground_bounding_box
from trackit.core.operator.numpy.bbox.validity import bbox_is_valid
from ... import TrackerEvaluationPipeline, TrackerEvaluationPipeline_Context


class SPMTrackTemplateFeatForegroundMaskGeneration(TrackerEvaluationPipeline):
    def __init__(self, template_size: Tuple[int, int], template_feat_size: Tuple[int, int], device: torch.device):
        self.template_size = template_size
        self.template_feat_size = template_feat_size
        self.stride = template_size[0] / template_feat_size[0], template_size[1] / template_feat_size[1]
        self.device = device
        self.background_value = 0
        self.foreground_value = 1

    def start(self, max_batch_size: int, global_objects: dict):
        self.memory_masks: Dict[Any, List[torch.Tensor]] = {}

    def stop(self, global_objects: dict):
        del self.memory_masks

    def _make_mask(self, bbox: np.ndarray, curation_parameter: np.ndarray) -> torch.Tensor:
        template_mask = torch.full((self.template_feat_size[1], self.template_feat_size[0]), self.background_value, dtype=torch.long)
        template_cropped_bbox = get_foreground_bounding_box(bbox, curation_parameter, self.stride)
        assert bbox_is_valid(template_cropped_bbox)
        template_cropped_bbox = torch.from_numpy(template_cropped_bbox)
        template_mask[template_cropped_bbox[1]: template_cropped_bbox[3], template_cropped_bbox[0]: template_cropped_bbox[2]] = self.foreground_value
        return template_mask

    def prepare_initialization(self, context: TrackerEvaluationPipeline_Context, model_input_params: dict):
        for task in context.input_data.tasks:
            if task.tracker_do_init_context is not None:
                current_init_context = task.tracker_do_init_context
                template_mask = self._make_mask(current_init_context.gt_bbox,
                                                current_init_context.input_data['curation_parameter'])
                self.memory_masks[task.id] = [template_mask]

    def prepare_tracking(self, context: TrackerEvaluationPipeline_Context, model_input_params: dict):
        reference_indices = context.temporary_objects.get('spmtrack_reference_indices')
        if reference_indices is None:
            return
        memory_lengths = context.temporary_objects['spmtrack_memory_lengths']
        task_ids = context.temporary_objects['task_ids']
        slots: List[List[torch.Tensor]] = [[], [], []]
        for task_id in task_ids:
            memory = self.memory_masks[task_id]
            if len(memory) != memory_lengths[task_id]:
                raise RuntimeError(f'Template and mask memories are out of step for task {task_id}: '
                                   f'{memory_lengths[task_id]} templates, {len(memory)} masks')
            for slot, memory_index in enumerate(reference_indices[task_id]):
                slots[slot].append(memory[memory_index].to(self.device))
        model_input_params.update({f'z_{slot}_feat_mask': torch.stack(slots[slot], dim=0) for slot in range(3)})

    def on_tracked(self, model_outputs, context: TrackerEvaluationPipeline_Context):
        for task_id, (bbox, curation_parameter) in context.temporary_objects.get('spmtrack_new_templates', {}).items():
            self.memory_masks[task_id].append(self._make_mask(bbox, curation_parameter))

    def end(self, context: TrackerEvaluationPipeline_Context):
        for task in context.input_data.tasks:
            if task.do_task_finalization:
                self.memory_masks.pop(task.id, None)
