"""Tracking pipeline for the SPMTrack baseline (independent of the STCMTrack `one_stream` pipeline).

Ported from the official
trackit/runner/evaluation/distributed/tracker_evaluator/default/pipelines/one_stream/__init__.py
@ WenRuiCai/SPMTrack c581fe27231f3e16c38578e47daddadfaf6ffd7d (Apache-2.0).

Behaviour kept from upstream
  * three templates per frame: the first-frame template and two historical references;
  * after every tracked frame a template is cropped around the predicted box (area factor 2.0,
    template resolution) and appended to the sequence memory, whatever the confidence;
  * reference selection, with n frames in memory:  n=1 -> [0,0,0], n=2 -> [0,1,1], n=3 -> [0,1,2],
    n>=4 -> [0, d//2, d + d//2] with d = n // 2   (upstream `select_memory_frames`, num_segments=2);
  * the search region comes from the previous predicted box (simple SiamFC cropping);
  * no CTR: no motion center correction and no residual-guided relocalization.

Deviations (engineering only):
  * the selection uses the memory length, which equals the upstream frame index when the sequence
    starts at frame 0 with no skipped update (upstream breaks with an index error otherwise);
  * templates and masks are indexed by the same selection computed once, and their lengths are
    checked; upstream recomputes the selection in two classes and indexes `tasks[i]` by the batch
    position of the tracking tasks (wrong when initialization and tracking tasks are mixed);
  * the template size and area factor come from the config instead of the literals 196 and 2.0;
  * a predicted box that is not valid after clipping to the image is not turned into a template
    (upstream would crash in the mask step); such skips are counted and printed;
  * sequence memory is released when the sequence ends and is reset when it is re-initialized
    (upstream keeps every frame of every finished sequence until the run ends);
  * non-finite model outputs raise instead of being replaced, like the shared evaluation core.
"""
from typing import Dict, Tuple, Callable, Any, Optional, List
import numpy as np
import torch
from dataclasses import dataclass, field
from trackit.core.operator.numpy.bbox.utility.image import bbox_clip_to_image_boundary_
from trackit.core.operator.numpy.bbox.validity import bbox_is_valid
from trackit.core.utils.siamfc_cropping import apply_siamfc_cropping, apply_siamfc_cropping_to_boxes, \
    reverse_siamfc_cropping_params, get_siamfc_cropping_params
from trackit.core.transforms.dataset_norm_stats import get_dataset_norm_stats_transform
from trackit.runner.evaluation.common.siamfc_search_region_cropping_params_provider import CroppingParameterProvider
from ....components.post_process import TrackerOutputPostProcess
from ....components.tensor_cache import CacheService, TensorCache

from ... import TrackerEvaluationPipeline

NUM_SEGMENTS = 2  # upstream `select_memory_frames`: num_segments = 2


def select_reference_indices(num_memory_frames: int) -> List[int]:
    """Indices into the sequence memory for the three template slots (z_0, z_1, z_2).

    memory[0] is the first-frame template; memory[k] is the template cropped from the prediction
    of tracked frame k.
    """
    if num_memory_frames < 1:
        raise ValueError('The sequence memory has no initial template')
    if num_memory_frames == 1:
        return [0, 0, 0]
    if num_memory_frames == 2:
        return [0, 1, 1]
    if num_memory_frames == 3:
        return [0, 1, 2]
    duration = num_memory_frames // NUM_SEGMENTS
    indexes = np.unique(np.concatenate([np.array([0]), np.arange(NUM_SEGMENTS) * duration + duration // 2]))
    assert len(indexes) == 3 and int(indexes[-1]) < num_memory_frames
    return [int(index) for index in indexes]


@dataclass
class _LocalContext:
    reset_frame_indices: List[int] = field(default_factory=list)
    siamfc_cropping_params_provider: Optional[CroppingParameterProvider] = None
    # Normalized template crops on the CPU; [0] is the initial template.
    memory_frames: List[torch.Tensor] = field(default_factory=list)


class SPMTrackOneStream_Evaluation_MainPipeline(TrackerEvaluationPipeline):
    def __init__(self, device: torch.device,
                 template_image_size: Tuple[int, int],
                 search_region_image_size: Tuple[int, int],  # W, H
                 search_curation_parameter_provider_factory: Callable[[], CroppingParameterProvider],
                 model_output_post_process: TrackerOutputPostProcess,
                 interpolation_mode: str, interpolation_align_corners: bool,
                 norm_stats_dataset_name: str, visualization: bool,
                 online_template_area_factor: float = 2.0):
        self.template_image_size = template_image_size
        self.search_region_image_size = search_region_image_size

        self.search_image_cropping_params_provider_factory = search_curation_parameter_provider_factory
        self.interpolation_mode = interpolation_mode
        self.interpolation_align_corners = interpolation_align_corners

        self.model_output_post_process = model_output_post_process
        self.device = device
        self.image_normalization_transform_ = get_dataset_norm_stats_transform(norm_stats_dataset_name, inplace=True)
        self.visualization = visualization
        self.online_template_area_factor = online_template_area_factor

    def start(self, max_batch_size: int, global_shared_objects):
        template_shape = (3, self.template_image_size[1], self.template_image_size[0])
        search_region_shape = (3, self.search_region_image_size[1], self.search_region_image_size[0])

        self.all_tracking_task_local_contexts: Dict[Any, _LocalContext] = {}
        self.all_tracking_template_cache = CacheService(max_batch_size,
                                                        TensorCache(max_batch_size, template_shape, self.device))
        self.all_tracking_template_image_mean_cache = CacheService(max_batch_size,
                                                                   TensorCache(max_batch_size, (3, ), self.device))
        global_shared_objects['template_cache'] = self.all_tracking_template_cache
        global_shared_objects['template_image_mean_cache'] = self.all_tracking_template_image_mean_cache

        self.cropping_parameter_cache = np.full((max_batch_size, 2, 2), float('nan'), dtype=np.float64)
        self.search_region_cache = torch.full((max_batch_size, *search_region_shape), float('nan'),
                                              dtype=torch.float, device=self.device)
        self.skipped_template_updates = 0
        self.model_output_post_process.start()

    def stop(self, global_shared_objects):
        self.model_output_post_process.stop()
        if self.skipped_template_updates > 0:
            print(f'SPMTrack pipeline: {self.skipped_template_updates} online template update(s) skipped '
                  f'because the predicted box was invalid after clipping.', flush=True)
        assert len(self.all_tracking_task_local_contexts) == 0, "bug check: some tracking sequences are not finished"
        del self.cropping_parameter_cache
        del self.search_region_cache
        del self.all_tracking_template_cache
        del self.all_tracking_template_image_mean_cache
        del self.all_tracking_task_local_contexts
        del self.skipped_template_updates

    def begin(self, context):
        for task in context.input_data.tasks:
            if task.task_creation_context is not None:
                assert task.id not in self.all_tracking_task_local_contexts
                self.all_tracking_task_local_contexts[task.id] = _LocalContext()

    def prepare_initialization(self, context, model_input_params):
        for task in context.input_data.tasks:
            if task.tracker_do_init_context is not None:
                init_context = task.tracker_do_init_context
                self.all_tracking_template_cache.put(task.id, init_context.input_data['curated_image'])
                self.all_tracking_template_image_mean_cache.put(task.id, init_context.input_data['image_mean'])
                cropping_params_provider = self.search_image_cropping_params_provider_factory()
                cropping_params_provider.initialize(init_context.gt_bbox)
                task_context = self.all_tracking_task_local_contexts[task.id]
                task_context.siamfc_cropping_params_provider = cropping_params_provider
                # (Re-)initialization starts a fresh memory with the initial template only.
                task_context.memory_frames = [init_context.input_data['curated_image'].detach().cpu().clone()]
                task_context.reset_frame_indices.append(init_context.frame_index)

    def prepare_tracking(self, context, model_input_params):
        num_tracking_sequence = 0
        task_ids = []
        image_size_list = []
        frame_indices = []
        for task in context.input_data.tasks:
            if task.tracker_do_tracking_context is not None:
                track_context = task.tracker_do_tracking_context
                template_image_mean = self.all_tracking_template_image_mean_cache.get(task.id)
                cropping_params_provider = self.all_tracking_task_local_contexts[task.id].siamfc_cropping_params_provider
                cropping_params = cropping_params_provider.get(np.array(self.search_region_image_size))
                x = track_context.input_data['image'].to(torch.float32)
                H, W = x.shape[-2:]
                image_size_list.append(np.array((W, H), dtype=np.int32))
                _, _, cropping_params = \
                    apply_siamfc_cropping(x, np.array(self.search_region_image_size), cropping_params,
                                          self.interpolation_mode, self.interpolation_align_corners,
                                          template_image_mean,
                                          out_image=self.search_region_cache[num_tracking_sequence, ...])
                self.cropping_parameter_cache[num_tracking_sequence, ...] = cropping_params
                num_tracking_sequence += 1
                task_ids.append(task.id)
                frame_indices.append(track_context.frame_index)

        if num_tracking_sequence == 0:
            return

        context.temporary_objects['task_ids'] = task_ids
        context.temporary_objects['x_frame_sizes'] = image_size_list
        context.temporary_objects['x_frame_indices'] = frame_indices
        context.temporary_objects['x_cropping_params'] = self.cropping_parameter_cache[: num_tracking_sequence, ...]

        x = self.search_region_cache[: num_tracking_sequence, ...]
        x = x / 255.
        self.image_normalization_transform_(x)

        # The selection is computed once; the mask plugin reads it, so z_i and z_i_feat_mask always
        # refer to the same remembered frames.
        reference_indices: Dict[Any, List[int]] = {}
        memory_lengths: Dict[Any, int] = {}
        template_slots: List[List[torch.Tensor]] = [[], [], []]
        for task_id in task_ids:
            memory = self.all_tracking_task_local_contexts[task_id].memory_frames
            indexes = select_reference_indices(len(memory))
            reference_indices[task_id] = indexes
            memory_lengths[task_id] = len(memory)
            for slot, memory_index in enumerate(indexes):
                template_slots[slot].append(memory[memory_index].to(x.device))
        context.temporary_objects['spmtrack_reference_indices'] = reference_indices
        context.temporary_objects['spmtrack_memory_lengths'] = memory_lengths

        model_input_params.update({'z_0': torch.stack(template_slots[0], dim=0),
                                   'z_1': torch.stack(template_slots[1], dim=0),
                                   'z_2': torch.stack(template_slots[2], dim=0),
                                   'x': x, 'ids': task_ids})

    def on_tracked(self, model_outputs, context):
        if model_outputs is None:
            return
        task_ids = context.temporary_objects['task_ids']
        x_frame_sizes = context.temporary_objects['x_frame_sizes']
        x_frame_indices = context.temporary_objects['x_frame_indices']
        x_cropping_params = context.temporary_objects['x_cropping_params']

        outputs = self.model_output_post_process(model_outputs)
        # shape: (num_tracking_sequence), dtype: torch.float
        all_predicted_score = outputs['confidence']
        # shape: (num_tracking_sequence, 4), dtype: torch.float
        all_predicted_bounding_box = outputs['box']

        assert all_predicted_score.ndim == 1
        assert all_predicted_bounding_box.ndim == 2
        assert all_predicted_bounding_box.shape[1] == 4
        assert len(task_ids) == len(all_predicted_score) == len(all_predicted_bounding_box)

        all_predicted_score = all_predicted_score.cpu()
        assert torch.all(torch.isfinite(all_predicted_score))
        all_predicted_bounding_box = all_predicted_bounding_box.cpu()
        assert torch.all(torch.isfinite(all_predicted_bounding_box))

        all_predicted_bounding_box = all_predicted_bounding_box.to(torch.float64)

        all_predicted_score = all_predicted_score.numpy()
        all_predicted_bounding_box = all_predicted_bounding_box.numpy()

        all_predicted_bounding_box_on_full_search_image = apply_siamfc_cropping_to_boxes(
            all_predicted_bounding_box, reverse_siamfc_cropping_params(x_cropping_params))
        for predicted_bounding_box_on_full_search_image, image_size in zip(
                all_predicted_bounding_box_on_full_search_image, x_frame_sizes):
            bbox_clip_to_image_boundary_(predicted_bounding_box_on_full_search_image, image_size)

        tracking_images = {task.id: task.tracker_do_tracking_context.input_data['image']
                           for task in context.input_data.tasks
                           if task.tracker_do_tracking_context is not None}
        new_templates: Dict[Any, Tuple[np.ndarray, np.ndarray]] = {}
        context.temporary_objects['spmtrack_new_templates'] = new_templates
        for index, (task_id, image_size, frame_index) in enumerate(zip(task_ids, x_frame_sizes, x_frame_indices)):
            predicted_score = all_predicted_score[index].item()
            predicted_bounding_box_on_full_search_image = all_predicted_bounding_box_on_full_search_image[index]
            local_task_context = self.all_tracking_task_local_contexts[task_id]
            local_task_context.siamfc_cropping_params_provider.update(predicted_score,
                                                                      predicted_bounding_box_on_full_search_image,
                                                                      image_size)
            context.result.submit(task_id,
                                  predicted_bounding_box_on_full_search_image,
                                  predicted_score,
                                  None)

            # Crop the current frame around the predicted box and keep it as a memory template.
            update = self._crop_online_template(tracking_images[task_id], predicted_bounding_box_on_full_search_image)
            if update is None:
                self.skipped_template_updates += 1
            else:
                new_template, template_curation_parameter = update
                local_task_context.memory_frames.append(new_template)
                new_templates[task_id] = (predicted_bounding_box_on_full_search_image.copy(), template_curation_parameter)

            if self.visualization:
                # Verbatim copy of the upstream module; importing the STCMTrack one_stream package would pull in CTR.
                from .visualization import visualize_tracking_result
                sequence_info = context.all_tracks[task_id].sequence_info
                x = self.search_region_cache[index, ...]
                predicted_bounding_box = all_predicted_bounding_box[index]
                visualize_tracking_result(sequence_info.dataset_name, sequence_info.sequence_name, frame_index,
                                          x, predicted_bounding_box, None, None)

        assert context.result.is_all_submitted()

    def _crop_online_template(self, tracked_image: torch.Tensor, bounding_box: np.ndarray):
        if not bbox_is_valid(bounding_box):
            return None
        template_size = np.array(self.template_image_size)
        template_curation_parameter = get_siamfc_cropping_params(bounding_box, self.online_template_area_factor,
                                                                 template_size)
        new_template, _, new_template_curation_parameter = apply_siamfc_cropping(
            tracked_image.to(torch.float32), template_size, template_curation_parameter,
            self.interpolation_mode, self.interpolation_align_corners)
        new_template.div_(255.)
        self.image_normalization_transform_(new_template)
        return new_template.cpu(), new_template_curation_parameter

    def end(self, context):
        for task in context.input_data.tasks:
            if task.do_task_finalization:
                self.all_tracking_template_cache.delete(task.id)
                self.all_tracking_template_image_mean_cache.delete(task.id)
                self.all_tracking_task_local_contexts.pop(task.id)
