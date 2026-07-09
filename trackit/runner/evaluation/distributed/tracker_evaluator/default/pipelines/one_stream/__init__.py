from typing import Dict, Tuple, Callable, Any, Optional, List, TYPE_CHECKING
import numpy as np
import torch
from dataclasses import dataclass, field
import cv2
from trackit.core.operator.numpy.bbox.utility.image import bbox_clip_to_image_boundary, bbox_clip_to_image_boundary_
from trackit.core.operator.numpy.bbox.validity import bbox_is_valid
from trackit.core.utils.siamfc_cropping import apply_siamfc_cropping, apply_siamfc_cropping_to_boxes, \
    reverse_siamfc_cropping_params, apply_siamfc_cropping_subpixel, scale_siamfc_cropping_params, get_siamfc_cropping_params
from trackit.core.transforms.dataset_norm_stats import get_dataset_norm_stats_transform
from trackit.runner.evaluation.common.siamfc_search_region_cropping_params_provider import CroppingParameterProvider
from ....components.post_process import TrackerOutputPostProcess
from ....components.tensor_cache import CacheService, TensorCache
from .motion_compensation import MotionCompensationModule

if TYPE_CHECKING:
    from ....components.segmentation import Segmentify_PostProcessor
else:
    Segmentify_PostProcessor = Any

from ... import TrackerEvaluationPipeline
@dataclass
class _LocalContext:
    reset_frame_indices: List[int] = field(default_factory=list)
    siamfc_cropping_params_provider: Optional[CroppingParameterProvider] = None
    last_valid_bbox: Optional[np.ndarray] = None


class OneStreamTracker_Evaluation_MainPipeline(TrackerEvaluationPipeline):
    memory_frame_limit = 64

    def __init__(self, device: torch.device,
                 template_image_size: Tuple[int, int],
                 search_region_image_size: Tuple[int, int],  # W, H
                 search_curation_parameter_provider_factory: Callable[[], CroppingParameterProvider],
                 model_output_post_process: TrackerOutputPostProcess,
                 segmentify_post_process: Optional[Segmentify_PostProcessor],
                 interpolation_mode: str, interpolation_align_corners: bool,
                 norm_stats_dataset_name: str, visualization: bool, model_type: str,
                 motion_compensation: Optional[MotionCompensationModule]):
        self.template_image_size = template_image_size
        self.search_region_image_size = search_region_image_size

        self.search_image_cropping_params_provider_factory = search_curation_parameter_provider_factory
        self.interpolation_mode = interpolation_mode
        self.interpolation_align_corners = interpolation_align_corners

        self.model_output_post_process = model_output_post_process
        self.segmentify_post_process = segmentify_post_process
        self.device = device
        self.image_normalization_transform_ = get_dataset_norm_stats_transform(norm_stats_dataset_name, inplace=True)
        self.visualization = visualization
        self.model_type = model_type
        self.motion_compensation = motion_compensation

    def _append_memory_frame(self, task_id, memory_frame: torch.Tensor):
        memory_frames = self.memory_frames.get(task_id)
        if memory_frames is None:
            self.memory_frames[task_id] = [memory_frame]
            return
        memory_frames.append(memory_frame)
        if len(memory_frames) <= self.memory_frame_limit:
            return
        # 保留首帧，并对后续历史做均匀采样，避免单序列历史无限增长。
        tail = memory_frames[1:]
        target_tail_size = self.memory_frame_limit - 1
        sample_indexes = np.linspace(0, len(tail) - 1, num=target_tail_size, dtype=np.int64)
        compacted = [memory_frames[0]]
        compacted.extend(tail[idx] for idx in sample_indexes.tolist())
        self.memory_frames[task_id] = compacted

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
        self.memory_frames = {}
        self.invalid_tracking_output_count = 0
        self.invalid_memory_bbox_count = 0
        self.model_output_post_process.start()
        if self.segmentify_post_process is not None:
            self.segmentify_post_process.start(max_batch_size)
        if self.motion_compensation is not None:
            self.motion_compensation.reset_statistics()

    def stop(self, global_shared_objects):
        if self.segmentify_post_process is not None:
            self.segmentify_post_process.stop()
        self.model_output_post_process.stop()
        if self.motion_compensation is not None:
            for summary_line in self.motion_compensation.format_summary():
                print(summary_line, flush=True)
            self.motion_compensation.clear()
        assert len(self.all_tracking_task_local_contexts) == 0, "bug check: some tracking sequences are not finished"
        del self.cropping_parameter_cache
        del self.search_region_cache
        del self.all_tracking_template_cache
        del self.all_tracking_template_image_mean_cache
        del self.all_tracking_task_local_contexts
        del self.memory_frames
        del self.invalid_tracking_output_count
        del self.invalid_memory_bbox_count

    def begin(self, context):
        for task in context.input_data.tasks:
            if task.task_creation_context is not None:
                assert task.id not in self.all_tracking_task_local_contexts
                self.all_tracking_task_local_contexts[task.id] = _LocalContext()

    def prepare_initialization(self, context, model_input_params):
        for task in context.input_data.tasks:
            if task.tracker_do_init_context is not None:
                init_context = task.tracker_do_init_context
                self.all_tracking_template_cache.put(task.id, init_context.input_data['curated_image']) # 记得记录memory
                self._append_memory_frame(task.id, init_context.input_data['curated_image'].detach().cpu().to(torch.float16))
                self.all_tracking_template_image_mean_cache.put(task.id, init_context.input_data['image_mean']) # 记得记录memory
                cropping_params_provider = self.search_image_cropping_params_provider_factory()
                cropping_params_provider.initialize(init_context.gt_bbox)
                task_context = self.all_tracking_task_local_contexts[task.id]
                task_context.siamfc_cropping_params_provider = cropping_params_provider
                task_context.last_valid_bbox = init_context.gt_bbox.astype(np.float64).copy()
                task_context.reset_frame_indices.append(init_context.frame_index)
                if self.motion_compensation is not None:
                    self.motion_compensation.reset(task.id, init_context.input_data.get('image'), init_context.gt_bbox)

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
                x = track_context.input_data['image'].to(torch.float32)
                H, W = x.shape[-2:]
                image_size = np.array((W, H), dtype=np.int32)
                image_size_list.append(image_size)
                if self.motion_compensation is not None:
                    compensated_bbox = self.motion_compensation.compensate_search_bbox(task.id, x, image_size)
                    if compensated_bbox is not None:
                        cropping_params_provider.update(None, compensated_bbox, image_size)
                cropping_params = cropping_params_provider.get(np.array(self.search_region_image_size))
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

        z = self.all_tracking_template_cache.get_batch(task_ids)
        x = self.search_region_cache[: num_tracking_sequence, ...]
        x = x / 255.
        self.image_normalization_transform_(x)

        model_input_params.update({'z_0': [], 'x': x, 'ids': task_ids})
        model_input_params.update({'z_1': []})
        model_input_params.update({'z_2': []})
        for i, task_id in enumerate(task_ids):
            if len(self.memory_frames[task_id]) == 1:
                model_input_params['z_0'].append(self.memory_frames[task_id][0].to(device=x.device, dtype=x.dtype))
                model_input_params['z_1'].append(self.memory_frames[task_id][0].to(device=x.device, dtype=x.dtype))
                model_input_params['z_2'].append(self.memory_frames[task_id][0].to(device=x.device, dtype=x.dtype))
            elif len(self.memory_frames[task_id]) == 2:
                model_input_params['z_0'].append(self.memory_frames[task_id][0].to(device=x.device, dtype=x.dtype))
                model_input_params['z_1'].append(self.memory_frames[task_id][1].to(device=x.device, dtype=x.dtype))
                model_input_params['z_2'].append(self.memory_frames[task_id][1].to(device=x.device, dtype=x.dtype))
            elif len(self.memory_frames[task_id]) == 3:
                model_input_params['z_0'].append(self.memory_frames[task_id][0].to(device=x.device, dtype=x.dtype))
                model_input_params['z_1'].append(self.memory_frames[task_id][1].to(device=x.device, dtype=x.dtype))
                model_input_params['z_2'].append(self.memory_frames[task_id][2].to(device=x.device, dtype=x.dtype))
            else:
                track_context = context.input_data.tasks[i].tracker_do_tracking_context
                assert task_id == context.input_data.tasks[i].id
                track_info = context.all_tracks[task_id]
                dataset = track_info.sequence_info.dataset_name
                seleted_indexes = self.select_memory_frames(len(self.memory_frames[task_id]), dataset)
                for i_selected, idx_selected in enumerate(seleted_indexes):
                    z_i = self.memory_frames[task_id][idx_selected].to(device=x.device, dtype=x.dtype)
                    model_input_params['z_' + str(i_selected)].append(z_i)

        model_input_params['z_0'] = torch.stack(model_input_params['z_0'], dim=0)
        model_input_params['z_1'] = torch.stack(model_input_params['z_1'], dim=0)
        model_input_params['z_2'] = torch.stack(model_input_params['z_2'], dim=0)

    def select_memory_frames(self, num_memory_frames, dataset_name):
        num_segments = 2
        assert num_memory_frames > num_segments
        if dataset_name == 'LaSOT' and 'B-378' in self.model_type:
            indexes = np.concatenate([
                np.array([0]),
                np.array([num_memory_frames // 3, 2 * num_memory_frames // 3])
            ])
        else:
            indexes = np.concatenate([
                np.array([0]),
                np.array([num_memory_frames // 4, 3 * num_memory_frames // 4])
            ])
        indexes = np.unique(indexes)
        return list(indexes)

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
        all_score_maps = outputs.get('score_map', None)
        all_box_maps = outputs.get('box_map', None)
        # shape: (num_tracking_sequence, H, W), dtype: torch.bool, allow None
        all_predicted_mask = outputs.get('mask', None)

        assert all_predicted_score.ndim == 1
        assert all_predicted_bounding_box.ndim == 2
        assert all_predicted_bounding_box.shape[1] == 4
        assert len(task_ids) == len(all_predicted_score) == len(all_predicted_bounding_box)
        if all_predicted_mask is not None:
            assert all_predicted_mask.ndim == 3
            assert all_predicted_mask.shape[0] == len(task_ids)

        all_predicted_score = all_predicted_score.cpu()
        all_predicted_bounding_box = all_predicted_bounding_box.cpu()
        if all_score_maps is not None:
            all_score_maps = all_score_maps.cpu().numpy()
        if all_box_maps is not None:
            all_box_maps = all_box_maps.cpu().numpy()
        finite_score = torch.isfinite(all_predicted_score)
        finite_bbox = torch.all(torch.isfinite(all_predicted_bounding_box), dim=1)
        finite_output = torch.logical_and(finite_score, finite_bbox)
        if not torch.all(finite_output):
            invalid_indexes = torch.nonzero(~finite_output, as_tuple=False).flatten().tolist()
            for invalid_index in invalid_indexes:
                self.invalid_tracking_output_count += 1
                if self.invalid_tracking_output_count <= 5:
                    print(f'warning: non-finite tracker output for task {task_ids[invalid_index]}: '
                          f'score={all_predicted_score[invalid_index].item()} '
                          f'box={all_predicted_bounding_box[invalid_index].tolist()}; '
                          f'reuse previous bbox',
                          flush=True)
            all_predicted_score = torch.nan_to_num(all_predicted_score, nan=0.0, posinf=0.0, neginf=0.0)
            all_predicted_bounding_box = torch.nan_to_num(
                all_predicted_bounding_box, nan=0.0, posinf=0.0, neginf=0.0)

        all_predicted_bounding_box = all_predicted_bounding_box.to(torch.float64)

        all_predicted_score = all_predicted_score.numpy()
        all_predicted_bounding_box = all_predicted_bounding_box.numpy()
        finite_output = finite_output.numpy()

        all_predicted_bounding_box_on_full_search_image = apply_siamfc_cropping_to_boxes(
            all_predicted_bounding_box, reverse_siamfc_cropping_params(x_cropping_params))
        for index, (predicted_bounding_box_on_full_search_image, image_size, task_id) in enumerate(zip(
                all_predicted_bounding_box_on_full_search_image, x_frame_sizes, task_ids)):
            bbox_clip_to_image_boundary_(predicted_bounding_box_on_full_search_image, image_size)
            if not finite_output[index] or not bbox_is_valid(predicted_bounding_box_on_full_search_image):
                if finite_output[index]:
                    self.invalid_tracking_output_count += 1
                local_task_context = self.all_tracking_task_local_contexts[task_id]
                fallback_bbox = local_task_context.last_valid_bbox
                if fallback_bbox is None or not bbox_is_valid(fallback_bbox):
                    fallback_bbox = np.array([0.0, 0.0, float(image_size[0]), float(image_size[1])],
                                             dtype=np.float64)
                fallback_bbox = bbox_clip_to_image_boundary(fallback_bbox, image_size)
                if self.invalid_tracking_output_count <= 5:
                    print(f'warning: invalid tracker bbox for task {task_id}: '
                          f'{predicted_bounding_box_on_full_search_image}; '
                          f'reuse {fallback_bbox}',
                          flush=True)
                all_predicted_bounding_box_on_full_search_image[index] = fallback_bbox

        tracking_images = {
            task.id: task.tracker_do_tracking_context.input_data['image']
            for task in context.input_data.tasks
            if task.tracker_do_tracking_context is not None
        }
        if self.motion_compensation is not None:
            for index, (task_id, image_size) in enumerate(zip(task_ids, x_frame_sizes)):
                corrected_bbox, _ = self.motion_compensation.correct_prediction(
                    task_id, tracking_images[task_id],
                    all_predicted_bounding_box_on_full_search_image[index],
                    all_predicted_score[index].item(), image_size,
                    tracker_score_map=all_score_maps[index] if all_score_maps is not None else None,
                    tracker_box_map=all_box_maps[index] if all_box_maps is not None else None,
                    cropping_params=x_cropping_params[index],
                    search_region_size=np.array(self.search_region_image_size, dtype=np.float64))
                all_predicted_bounding_box_on_full_search_image[index] = corrected_bbox
                if not bbox_is_valid(all_predicted_bounding_box_on_full_search_image[index]):
                    local_task_context = self.all_tracking_task_local_contexts[task_id]
                    all_predicted_bounding_box_on_full_search_image[index] = bbox_clip_to_image_boundary(
                        local_task_context.last_valid_bbox, image_size)

        all_predicted_mask_on_full_search_image = None
        if all_predicted_mask is not None:
            all_predicted_mask_on_full_search_image = []
            for curr_mask, curr_image_size, curr_cropping_parameter in zip(
                    all_predicted_mask, x_frame_sizes, x_cropping_params):
                mask_h, mask_w = curr_mask.shape
                curr_cropping_parameter = scale_siamfc_cropping_params(curr_cropping_parameter,
                                                                       np.array(self.search_region_image_size),
                                                                       np.array((mask_w, mask_h)))
                predicted_mask_on_full_search_image = apply_siamfc_cropping_subpixel(
                    curr_mask.to(torch.float32).unsqueeze(0),
                    np.array(curr_image_size), reverse_siamfc_cropping_params(curr_cropping_parameter),
                    self.interpolation_mode, self.interpolation_align_corners)
                all_predicted_mask_on_full_search_image.append(
                    predicted_mask_on_full_search_image.squeeze(0).to(torch.bool).cpu().numpy())
        else:
            if self.segmentify_post_process is not None:
                full_search_region_images = []

                for task in context.input_data.tasks:
                    if task.tracker_do_tracking_context is not None:
                        full_search_region_images.append(task.tracker_do_tracking_context.input_data['image'])
                all_predicted_mask_on_full_search_image = (
                    self.segmentify_post_process(full_search_region_images,
                                                 all_predicted_bounding_box_on_full_search_image))
        context.temporary_objects['memory_new_z_curated'] = {}
        context.temporary_objects['memory_new_z_curated_norm'] = {}
        context.temporary_objects['memory_new_z_curated_bbox'] = {}
        context.temporary_objects['memory_new_z_curation_parameter'] = {}
        for index, (task_id, image_size, frame_index) in enumerate(zip(task_ids, x_frame_sizes, x_frame_indices)):
            predicted_score = all_predicted_score[index].item()
            predicted_bounding_box_on_full_search_image = all_predicted_bounding_box_on_full_search_image[index] # 推理的预测bbox
            local_task_context = self.all_tracking_task_local_contexts[task_id]
            local_task_context.siamfc_cropping_params_provider.update(predicted_score,
                                                                      predicted_bounding_box_on_full_search_image,
                                                                      image_size)
            if bbox_is_valid(predicted_bounding_box_on_full_search_image):
                local_task_context.last_valid_bbox = predicted_bounding_box_on_full_search_image.copy()
            if self.motion_compensation is not None:
                self.motion_compensation.update(task_id, tracking_images[task_id],
                                                predicted_bounding_box_on_full_search_image, image_size)
            predicted_mask_on_full_search_image = all_predicted_mask_on_full_search_image[index] \
                if all_predicted_mask_on_full_search_image is not None else None
            context.result.submit(task_id,
                                  predicted_bounding_box_on_full_search_image,
                                  predicted_score,
                                  predicted_mask_on_full_search_image)

            # 裁剪当前frame留作memory
            tracked_img = tracking_images[task_id]
            memory_bounding_box_on_full_search_image = predicted_bounding_box_on_full_search_image
            if not bbox_is_valid(memory_bounding_box_on_full_search_image):
                self.invalid_memory_bbox_count += 1
                if self.invalid_memory_bbox_count <= 5:
                    print(f'warning: invalid memory bbox for task {task_id}: '
                          f'{memory_bounding_box_on_full_search_image}; reuse previous memory',
                          flush=True)
                previous_memory = self.memory_frames[task_id][-1].clone()
                self._append_memory_frame(task_id, previous_memory)
                context.temporary_objects['memory_new_z_curated'][task_id] = previous_memory.clone()
                context.temporary_objects['memory_new_z_curated_norm'][task_id] = previous_memory
                context.temporary_objects['memory_new_z_curated_bbox'][task_id] = memory_bounding_box_on_full_search_image
                context.temporary_objects['memory_new_z_curation_parameter'][task_id] = np.stack(
                    (np.ones(2, dtype=np.float64), np.zeros(2, dtype=np.float64)))
                continue
            template_curation_parameter = get_siamfc_cropping_params(memory_bounding_box_on_full_search_image, 2.0, np.array([196, 196])) # 2, [196,196]
            new_z_curated, new_z_image_mean, new_template_curation_parameter = apply_siamfc_cropping(
                tracked_img.to(torch.float32), np.array([196, 196]), template_curation_parameter,
                self.interpolation_mode, self.interpolation_align_corners) # bilinear, False
            #cv2.imwrite("1.jpg", new_z_curated.permute(1,2,0).int().cpu().numpy())
            context.temporary_objects['memory_new_z_curated'][task_id] = new_z_curated.clone()
            new_z_curated.div_(255.)
            self.image_normalization_transform_(new_z_curated)
            self._append_memory_frame(task_id, new_z_curated.detach().cpu().to(torch.float16))
            context.temporary_objects['memory_new_z_curated_norm'][task_id] = new_z_curated
            context.temporary_objects['memory_new_z_curated_bbox'][task_id] = memory_bounding_box_on_full_search_image
            context.temporary_objects['memory_new_z_curation_parameter'][task_id] = new_template_curation_parameter

            if self.visualization:
                from .visualization import visualize_tracking_result
                sequence_info = context.all_tracks[task_id].sequence_info
                x = self.search_region_cache[index, ...]
                predicted_bounding_box = all_predicted_bounding_box[index]
                predicted_mask = all_predicted_mask[index] if all_predicted_mask is not None else None
                visualize_tracking_result(sequence_info.dataset_name, sequence_info.sequence_name, frame_index,
                                          x, predicted_bounding_box,
                                          predicted_mask, predicted_mask_on_full_search_image)

        assert context.result.is_all_submitted()

    def end(self, context):
        for task in context.input_data.tasks:
            if task.do_task_finalization:
                self.all_tracking_template_cache.delete(task.id)
                self.all_tracking_template_image_mean_cache.delete(task.id)
                self.all_tracking_task_local_contexts.pop(task.id)
                if self.motion_compensation is not None:
                    self.motion_compensation.forget(task.id)
