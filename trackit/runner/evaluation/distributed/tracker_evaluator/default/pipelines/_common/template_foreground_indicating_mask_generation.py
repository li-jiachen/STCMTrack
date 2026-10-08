from typing import Tuple
import torch

from trackit.core.utils.bbox_mask_gen import get_foreground_bounding_box
from trackit.core.operator.numpy.bbox.validity import bbox_is_valid
from trackit.runner.evaluation.distributed.tracker_evaluator.default.pipelines import TrackerEvaluationPipeline_Context

from ....default import TrackerEvaluationPipeline, TrackerEvaluationPipeline_Context
from ....components.tensor_cache import CacheService, TensorCache

class TemplateFeatForegroundMaskGeneration(TrackerEvaluationPipeline):
    def __init__(self, template_size: Tuple[int, int], template_feat_size: Tuple[int, int], device: torch.device, model_type: str,
                 provide_during_tracking: bool = True):
        self.template_size = template_size
        self.template_feat_size = template_feat_size
        self.stride = template_size[0] / template_feat_size[0], template_size[1] / template_feat_size[1]
        self.device = device
        self.background_value = 0
        self.foreground_value = 1
        self.provide_during_tracking = provide_during_tracking
        self.model_type = model_type

    def start(self, max_batch_size: int, global_objects: dict):
        self.template_mask_cache = CacheService(max_batch_size, TensorCache(
            max_batch_size, (self.template_feat_size[1], self.template_feat_size[0]), self.device, torch.long))

    def stop(self, global_objects: dict):
        del self.template_mask_cache

    def prepare_initialization(self, context: TrackerEvaluationPipeline_Context, model_input_params: dict):
        task_ids = []
        for task in context.input_data.tasks:
            init = task.tracker_do_init_context
            if init is None:
                continue
            mask = torch.full((self.template_feat_size[1], self.template_feat_size[0]),
                              self.background_value, dtype=torch.long)
            bbox = get_foreground_bounding_box(init.gt_bbox, init.input_data['curation_parameter'], self.stride)
            if not bbox_is_valid(bbox):
                raise ValueError('The first-frame template must contain a valid target')
            mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = self.foreground_value
            self.template_mask_cache.put(task.id, mask.to(self.device))
            task_ids.append(task.id)
        if not self.provide_during_tracking and task_ids:
            model_input_params['z_0_feat_mask'] = self.template_mask_cache.get_batch(task_ids)

    def prepare_tracking(self, context: TrackerEvaluationPipeline_Context, model_input_params: dict):
        task_ids = [task.id for task in context.input_data.tasks if task.tracker_do_tracking_context is not None]
        if self.provide_during_tracking and task_ids:
            model_input_params['z_0_feat_mask'] = self.template_mask_cache.get_batch(task_ids)

    def end(self, context: TrackerEvaluationPipeline_Context):
        for task in context.input_data.tasks:
            if task.do_task_finalization:
                self.template_mask_cache.delete(task.id)
