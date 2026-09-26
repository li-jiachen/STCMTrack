from typing import Sequence, Optional
import numpy as np
from trackit.data.source import TrackingDataset
from trackit.data.sampling.per_sequence import RandomAccessiblePerSequenceSampler

from ._algos import get_random_positive_siamese_training_pair_from_track, _get_random_track
from ._types import SiamesePairSamplingMethod, SiamesePairNegativeSamplingMethod, SiameseTrainingPairMultiSamplingResult, SamplingResult_Element, SiameseTrainingPairSamplingResult
from ._distractor import DistractorGenerator


def _get_frame_size(frame):
    return np.asarray(frame.get_frame_size(), dtype=np.float64)


def _scale_bbox_between_frames(bbox: np.ndarray, source_frame, target_frame):
    source_frame_size = _get_frame_size(source_frame)
    target_frame_size = _get_frame_size(target_frame)
    if np.array_equal(source_frame_size, target_frame_size):
        return bbox
    scale = np.repeat(target_frame_size / source_frame_size, 2)
    return bbox * scale


def _get_negative_search_bbox(track, visible_indices: np.ndarray, absent_frame_index: int,
                              rng_engine: np.random.Generator):
    reference_frame_index = rng_engine.choice(visible_indices)
    reference_frame = track[reference_frame_index]
    absent_frame = track[absent_frame_index]
    bbox = reference_frame.get_bounding_box().astype(np.float64)
    bbox = _scale_bbox_between_frames(bbox, reference_frame, absent_frame)
    return tuple(float(value) for value in bbox)


class SiamFCTrainingPairSampler:
    def __init__(self, datasets: Sequence[TrackingDataset],
                 dataset_weights: np.ndarray,
                 sequence_picker: Optional[RandomAccessiblePerSequenceSampler],
                 siamese_sampling_frame_range: int,
                 siamese_sampling_method: SiamesePairSamplingMethod,
                 siamese_sampling_frame_range_auto_extend_step: int,
                 siamese_sampling_frame_range_auto_extend_max_retry_count: int,
                 siamese_sampling_disable_frame_range_constraint_if_search_frame_not_found: bool,
                 negative_sample_weight: float,
                 negative_sample_generation_methods: Sequence[SiamesePairNegativeSamplingMethod],
                 negative_sample_generation_method_weights: Optional[np.ndarray],
                 strict_negative_sample: bool,
                 negative_sample_max_retry_count: int,
                 num_template_frames: int,
                 num_search_frames: int,
                 max_sample_interval: int):
        self.datasets = datasets

        dataset_weights = np.array(dataset_weights, dtype=np.float64)
        dataset_weights /= dataset_weights.sum()
        self.dataset_weights = dataset_weights

        self.sequence_picker = sequence_picker
        self.siamese_sampling_frame_range = siamese_sampling_frame_range
        self.siamese_sampling_method = siamese_sampling_method
        self.siamese_sampling_frame_range_auto_extend_step = siamese_sampling_frame_range_auto_extend_step
        self.siamese_sampling_frame_range_auto_extend_max_retry_count = siamese_sampling_frame_range_auto_extend_max_retry_count
        self.siamese_sampling_disable_frame_range_constraint_if_search_frame_not_found = siamese_sampling_disable_frame_range_constraint_if_search_frame_not_found
        self.negative_sample_weight = negative_sample_weight
        self.negative_sample_generation_methods = negative_sample_generation_methods
        self.negative_sample_generation_method_weights = negative_sample_generation_method_weights
        self.strict_negative_sample = strict_negative_sample
        self.negative_sample_max_retry_count = negative_sample_max_retry_count
        self.max_sample_interval = max_sample_interval
        self.num_template_frames = num_template_frames
        self.num_search_frames = num_search_frames

        if len(negative_sample_generation_methods) > 0:
            distractor_picker_required = False
            for weight, method in zip(negative_sample_generation_method_weights, negative_sample_generation_methods):
                if weight > 0 and method == SiamesePairNegativeSamplingMethod.distractor:
                    distractor_picker_required = True
                    break
            if distractor_picker_required:
                self.distractor_pickers = tuple(DistractorGenerator(dataset) for dataset in self.datasets)

    def __call__(self, index: Optional[int], rng_engine: np.random.Generator) -> SiameseTrainingPairSamplingResult:
        dataset_index, sequence_index, sequence, track = self._sample_sequence_and_track(index, rng_engine)

        if self.negative_sample_weight > 0:
            is_positive = rng_engine.random() > self.negative_sample_weight
        else:
            is_positive = True

        if is_positive:
            return self._make_positive_pair(dataset_index, sequence_index, track, rng_engine)

        negative_pair = self._make_negative_pair(dataset_index, sequence_index, track, rng_engine)
        if negative_pair is not None:
            return negative_pair

        if self.strict_negative_sample:
            raise RuntimeError(
                f'Cannot sample a strict negative training pair after '
                f'{self.negative_sample_max_retry_count} retries. '
                f'Check whether the training set contains absent-frame annotations.')

        return self._make_positive_pair(dataset_index, sequence_index, track, rng_engine)

    def _sample_sequence_and_track(self, index: Optional[int], rng_engine: np.random.Generator):
        if index is not None:
            assert self.sequence_picker is not None, 'Sequence picker is required for indexed sampling'
            dataset_index, sequence_index = self.sequence_picker[index]
            dataset = self.datasets[dataset_index]
            sequence = dataset[sequence_index]
        else:
            dataset_index = rng_engine.choice(np.arange(len(self.datasets)), p=self.dataset_weights)
            dataset = self.datasets[dataset_index]
            sequence_index = rng_engine.integers(0, len(dataset))
            sequence = dataset[sequence_index]

        track = _get_random_track(sequence, rng_engine)
        return dataset_index, sequence_index, sequence, track

    def _make_positive_pair(self, dataset_index: int, sequence_index: int, track, rng_engine: np.random.Generator):
        template_indices, search_indices = get_random_positive_siamese_training_pair_from_track(
            track, self.siamese_sampling_frame_range, self.siamese_sampling_method, rng_engine,
            self.siamese_sampling_frame_range_auto_extend_step,
            self.siamese_sampling_frame_range_auto_extend_max_retry_count,
            self.siamese_sampling_disable_frame_range_constraint_if_search_frame_not_found,
            self.num_template_frames, self.num_search_frames, self.max_sample_interval
        )

        return SiameseTrainingPairMultiSamplingResult(
            [SamplingResult_Element(dataset_index, sequence_index, track.get_object_id(), frame_indices) for frame_indices in template_indices],
            [SamplingResult_Element(dataset_index, sequence_index, track.get_object_id(), frame_indices) for frame_indices in search_indices],
            True)

    def _make_negative_pair(self, dataset_index: int, sequence_index: int, track,
                            rng_engine: np.random.Generator):
        for attempt in range(max(1, self.negative_sample_max_retry_count)):
            existence_flags = track.get_all_object_existence_flag()
            if existence_flags is not None:
                existence_flags = np.asarray(existence_flags, dtype=np.bool_)
            if existence_flags is not None and np.any(existence_flags) and np.any(~existence_flags):
                visible_indices = np.flatnonzero(existence_flags)
                absent_indices = np.flatnonzero(~existence_flags)
                template_indices = rng_engine.choice(visible_indices, size=self.num_template_frames,
                                                     replace=len(visible_indices) < self.num_template_frames)
                search_indices = rng_engine.choice(absent_indices, size=self.num_search_frames,
                                                   replace=len(absent_indices) < self.num_search_frames)
                search_bboxes = [
                    _get_negative_search_bbox(track, visible_indices, frame_index, rng_engine)
                    for frame_index in search_indices
                ]
                return SiameseTrainingPairMultiSamplingResult(
                    [SamplingResult_Element(dataset_index, sequence_index, track.get_object_id(), frame_index) for frame_index in template_indices],
                    [SamplingResult_Element(dataset_index, sequence_index, track.get_object_id(), frame_index, bbox) for frame_index, bbox in zip(search_indices, search_bboxes)],
                    False)

            if attempt + 1 < self.negative_sample_max_retry_count:
                dataset_index, sequence_index, _, track = self._sample_sequence_and_track(None, rng_engine)

        return None
