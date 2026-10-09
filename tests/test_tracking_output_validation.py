"""Shared failure handling for the independent SPMTrack and STCMTrack pipelines."""
from collections import deque
from types import SimpleNamespace
import contextlib
import io
import unittest
from unittest.mock import Mock

import numpy as np
import torch

from trackit.core.evaluation.antiuav import evaluate_sequence
from trackit.core.utils.siamfc_cropping import get_siamfc_cropping_params
from trackit.runner.evaluation.common.siamfc_search_region_cropping_params_provider.simple import (
    SiamFCCroppingParameterSimpleProvider)
from trackit.runner.evaluation.distributed.tracker_evaluator.components.post_process.box_with_score_map import (
    PostProcessing_BoxWithScoreMap)
from trackit.runner.evaluation.distributed.tracker_evaluator.components.post_process.spmtrack_box_with_score_map import (
    PostProcessing_BoxWithScoreMap_SPMTrack)
from trackit.runner.evaluation.distributed.tracker_evaluator.default.pipelines.one_stream import (
    OneStreamTracker_Evaluation_MainPipeline)
from trackit.runner.evaluation.distributed.tracker_evaluator.default.pipelines.spmtrack_one_stream import (
    SPMTrackOneStream_Evaluation_MainPipeline)


POST_PROCESSORS = (PostProcessing_BoxWithScoreMap, PostProcessing_BoxWithScoreMap_SPMTrack)
PIPELINES = (OneStreamTracker_Evaluation_MainPipeline, SPMTrackOneStream_Evaluation_MainPipeline)


def dense_output():
    return {'score_map': torch.tensor([[[3., 0.], [0., 0.]]]),
            'boxes': torch.tensor([[[[.1, .1, .8, .8]] * 2] * 2])}


def pipeline_fixture(cls, prediction):
    pipeline = cls.__new__(cls)
    pipeline.model_output_post_process = lambda _: prediction
    pipeline.ctr = None
    pipeline.segmentify_post_process = None
    pipeline.visualization = False
    pipeline.invalid_tracking_output_count = 0
    pipeline.skipped_template_updates = 0
    box = np.array([20., 20., 40., 40.])
    crop = SiamFCCroppingParameterSimpleProvider(5., 1.)
    crop.initialize(box)
    pipeline.all_tracking_task_local_contexts = {
        91: SimpleNamespace(siamfc_cropping_params_provider=crop, memory_frames=deque())}
    task = SimpleNamespace(id=91, tracker_do_tracking_context=SimpleNamespace(
        input_data={'image': torch.ones(3, 64, 64)}))
    context = SimpleNamespace(
        temporary_objects={'task_ids': [91], 'x_frame_sizes': [np.array([64, 64])],
                           'x_frame_indices': [1], 'x_cropping_params': np.array([
                               get_siamfc_cropping_params(box, 5., np.array([28, 28]))])},
        input_data=SimpleNamespace(tasks=[task]),
        result=SimpleNamespace(submit=Mock(), is_all_submitted=lambda: True))
    return pipeline, context, crop, box


class TrackingOutputValidationTests(unittest.TestCase):
    def test_nonfinite_dense_outputs_cannot_be_hidden_by_sigmoid_or_peak_selection(self):
        for cls in POST_PROCESSORS:
            post_process = cls(torch.device('cpu'), (2, 2), (28, 28), 0.)
            post_process.start()
            try:
                for field in ('score_map', 'boxes'):
                    for value in (float('nan'), float('inf'), -float('inf')):
                        with self.subTest(model=cls.__name__, field=field, value=value):
                            output = dense_output()
                            # The last box is not at the selected peak; it must still be checked.
                            output[field].reshape(-1)[-1] = value
                            with self.assertRaisesRegex(ValueError, f'non-finite.*{field}'):
                                post_process(output)
            finally:
                post_process.stop()

    def test_finite_outputs_keep_the_shared_zero_window_box_selection(self):
        boxes = []
        for cls in POST_PROCESSORS:
            post_process = cls(torch.device('cpu'), (2, 2), (28, 28), 0.)
            post_process.start()
            try:
                output = post_process(dense_output())
                self.assertTrue(torch.isfinite(output['confidence']).all())
                boxes.append(output['box'])
            finally:
                post_process.stop()
        torch.testing.assert_close(boxes[0], boxes[1], rtol=0, atol=0)

    def test_nonfinite_selected_predictions_stop_both_pipelines_before_submission(self):
        for cls in PIPELINES:
            for field in ('confidence', 'box'):
                for value in (float('nan'), float('inf'), -float('inf')):
                    with self.subTest(model=cls.__name__, field=field, value=value):
                        prediction = {'confidence': torch.tensor([.5]),
                                      'box': torch.tensor([[1., 1., 20., 20.]])}
                        prediction[field].reshape(-1)[0] = value
                        pipeline, context, _, _ = pipeline_fixture(cls, prediction)
                        with self.assertRaisesRegex(ValueError, 'non-finite'):
                            pipeline.on_tracked({}, context)
                        context.result.submit.assert_not_called()

    def test_invalid_finite_final_box_is_a_failure_and_preserves_valid_crop_state(self):
        for cls in PIPELINES:
            with self.subTest(model=cls.__name__):
                prediction = {'confidence': torch.tensor([.5]), 'box': torch.zeros(1, 4)}
                pipeline, context, crop, original_box = pipeline_fixture(cls, prediction)
                with contextlib.redirect_stdout(io.StringIO()):
                    pipeline.on_tracked({}, context)
                context.result.submit.assert_called_once()
                submitted = context.result.submit.call_args.args[1].copy()
                submitted[2:] -= submitted[:2]
                metrics = evaluate_sequence(submitted[None], np.array([[20., 20., 20., 20.]]), [1])
                self.assertEqual((metrics.auc, metrics.p20, metrics.pn), (0., 0., 0.))
                np.testing.assert_array_equal(crop.cached_bbox, original_box)


if __name__ == '__main__':
    unittest.main()
