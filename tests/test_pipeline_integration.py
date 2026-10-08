"""Integration coverage: real preprocessing, loss, evaluation pipeline and YAML mixins."""
import copy
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch

from test_stcmtrack import make_model
from trackit.core.boot.funcs.utils.custom_yaml_loader import load_yaml
from trackit.core.boot.funcs.mixin import apply_static_mixin_rules
from trackit.data.methods.siamese_tracker_train._types import SiameseTrainingMultiPair, SOTFrameInfo
from trackit.data.methods.siamese_tracker_train.transform.default.processor import (
    SiamFCCroppingParameter, SiamTrackerTrainingPairProcessor, SiamTrackerTrainingPairProcessorBatchCollator)
from trackit.data.methods.siamese_tracker_train.transform.default.augmentation.builder import build_augmentation_pipeline
from trackit.data.methods.siamese_tracker_train.transform.default.plugin.box_with_score_map_label_gen import (
    BoxWithScoreMapLabelGenerator, box_with_score_map_label_collator)
from trackit.data.methods.siamese_tracker_train.transform.default.plugin.template_foreground_indicating_mask_gen import (
    TemplateFeatMaskGenerator, template_feat_mask_data_collator)
from trackit.data.protocol.train_input import TrainData
from trackit.criteria.methods.box_with_score_map.builder import build_box_with_score_map_criteria
from trackit.data.protocol import SequenceInfo
from trackit.data.protocol.eval_input import TrackerEvalData, TrackerEvalData_TaskDesc, TrackerEvalData_FrameData
from trackit.runner.evaluation.distributed.tracker_evaluator import run_tracker_evaluator
from trackit.runner.evaluation.distributed.tracker_evaluator.default import DefaultTrackerEvaluator
from trackit.runner.evaluation.distributed.tracker_evaluator.default.pipelines.one_stream.builder import build_one_stream_tracker_pipeline
from trackit.core.utils.siamfc_cropping import get_siamfc_cropping_params, apply_siamfc_cropping
from trackit.core.transforms.dataset_norm_stats import get_dataset_norm_stats_transform
from trackit.core.evaluation.antiuav import evaluate_sequence
from trackit.models.methods.STCMTrack.modules.ltcp import LTCPConfig
from trackit.runner.evaluation.distributed.tracker_evaluator.default.pipelines.one_stream.ctr import CTRConfig, _to_gray_uint8

ROOT = Path(__file__).resolve().parents[1]


def config():
    return load_yaml(str(ROOT / 'config/STCMTrack/dinov2/config.yaml'))


class IntegrationTests(unittest.TestCase):
    def test_all_dataset_variant_configs(self):
        # These are the STCMTrack-method variants. `baseline` / `spmtrack` build SPMTrack (config/SPMTrack) and are
        # covered by tests/test_spmtrack_static.py; `stcm_base` is STCMTrack with every component off.
        variants = {'stcm_base': (), 'ltcp': ('ltcp',), 'mcc': ('ctr', 'ctr_no_rgtc'),
                    'rgtc': ('ctr', 'ctr_no_mcc'), 'ltcp_mcc': ('ltcp', 'ctr', 'ctr_no_rgtc'),
                    'ltcp_rgtc': ('ltcp', 'ctr', 'ctr_no_mcc'), 'mcc_rgtc': ('ctr',), 'full': ('ltcp', 'ctr')}
        for dataset in ('antiuav410', 'antiuav300'):
            for variant, mixins in variants.items():
                with self.subTest(dataset=dataset, variant=variant):
                    cfg = config()
                    ordered = (('dataset_antiuav300',) if dataset == 'antiuav300' else ()) + mixins + ('evaluation',)
                    for name in ordered:
                        apply_static_mixin_rules(load_yaml(str(ROOT / f'config/STCMTrack/_mixin/{name}.yaml')), cfg)
                    LTCPConfig.from_dict(cfg['model'].get('ltcp'))
                    ctr = cfg['run']['runner']['test']['evaluator']['pipeline'].get('ctr')
                    if ctr is not None:
                        CTRConfig.from_dict(ctr)
                        self.assertTrue(cfg['run']['data']['eval']['transform']['with_full_template_image'])
                    self.assertEqual(cfg['run']['runner']['test']['evaluator']['pipeline']['post_process']['window_penalty'], 0.)
                    self.assertEqual(set(cfg['run']['data']), {'eval'})
                    self.assertNotIn('filters', cfg['run']['data']['eval']['source']['parameters']['datasets'][0])
                    if 'ltcp' in mixins:
                        self.assertIn('-ltcp', cfg['name'])
        cfg = config()
        for name in ('ltcp', 'ltcp_stage2'):
            apply_static_mixin_rules(load_yaml(str(ROOT / f'config/STCMTrack/_mixin/{name}.yaml')), cfg)
        self.assertEqual(cfg['run']['num_epochs'], 20)
        self.assertTrue(cfg['model']['ltcp']['train_only'])
        self.assertEqual(config()['run']['num_epochs'], 80)
        self.assertFalse(config()['run']['runner']['train']['torch_compile']['enabled'])

    def test_preprocessing_collation_bce_giou_backward(self):
        cfg = config()
        aug = cfg['run']['data']['train']['transform']['augmentation']
        rng = np.random.default_rng(21)
        image = rng.integers(0, 256, (80, 80, 3), dtype=np.uint8)
        frame = lambda: SOTFrameInfo(lambda: image.copy(), np.array([30., 30., 50., 50.]), True, None, None, None)
        pair = SiameseTrainingMultiPair(True, [frame()], [frame(), frame(), frame()])
        processor = SiamTrackerTrainingPairProcessor(
            SiamFCCroppingParameter(np.array([14, 14]), 2.),
            SiamFCCroppingParameter(np.array([28, 28]), 5., .25, .5),
            build_augmentation_pipeline(aug), 'imagenet',
            [BoxWithScoreMapLabelGenerator((2, 2), (28, 28), 'center', 0), TemplateFeatMaskGenerator((14, 14), (1, 1))],
            temporal_consistent_crops=True)
        sample = processor(pair, rng)
        self.assertIsNotNone(sample)
        for i in range(1, 3):
            torch.testing.assert_close(sample['x_0_cropped_image'], sample[f'x_{i}_cropped_image'], rtol=0, atol=0)
        batch = TrainData({}, {'epoch': 0}, {})
        SiamTrackerTrainingPairProcessorBatchCollator([box_with_score_map_label_collator, template_feat_mask_data_collator])([sample], batch)
        self.assertEqual(set(batch.input), {'z_0', 'z_0_feat_mask', 'x_0', 'x_1', 'x_2'})
        for i in range(3):
            self.assertEqual(batch.target[f'num_positive_samples_{i}'].item(), 1)
        model = make_model(stage2=True)
        with contextlib.redirect_stdout(io.StringIO()):
            criterion = build_box_with_score_map_criteria(cfg['run']['runner']['train']['criteria'])
        output = model(**batch.input)
        result = criterion(output, batch.target)
        self.assertTrue(torch.isfinite(result.loss).all())
        expected = sum(result.metrics.values()) / 3
        self.assertAlmostEqual(result.loss.item(), expected, places=5)
        result.loss.backward()
        self.assertTrue(torch.isfinite(model.ltcp.gate.weight.grad).all())

    def test_70_frame_pipeline_keeps_first_frame_template_and_releases_sequence(self):
        cfg = config()
        cfg['common'].update(template_size=[14, 14], search_region_size=[28, 28],
                             template_feat_size=[1, 1], search_region_feat_size=[2, 2], response_map_size=[2, 2])
        pipeline_cfg = cfg['run']['runner']['test']['evaluator']['pipeline']
        pipeline_cfg['ctr'] = {'enabled': True, 'print_summary': False}
        with contextlib.redirect_stdout(io.StringIO()):
            pipelines = build_one_stream_tracker_pipeline(pipeline_cfg, cfg, torch.device('cpu'))
        evaluator = DefaultTrackerEvaluator(pipelines)
        with patch('trackit.runner.evaluation.distributed.tracker_evaluator.default.get_current_data_context',
                   return_value=SimpleNamespace(variables={'batch_size': 1, 'num_workers': 1})):
            evaluator.on_epoch_begin()
        raw = make_model(True).eval(); raw.init_eval(1)
        image = torch.full((3, 64, 64), 70.); image[:, 20:40, 20:40] = 200.
        box = np.array([20., 20., 40., 40.]); curation = get_siamfc_cropping_params(box, 2., np.array([14, 14]))
        template, mean, curation = apply_siamfc_cropping(image, np.array([14, 14]), curation, 'bilinear', False)
        template.div_(255.); get_dataset_norm_stats_transform('imagenet', inplace=True)(template)
        init = TrackerEvalData_FrameData(0, box, None, {'image': image, 'curated_image': template,
                                                     'image_mean': mean, 'curation_parameter': curation})
        sequence = SequenceInfo('ANTIUAV410', ('test',), 'synthetic', 's', 71, None)
        calls = []
        def infer(params):
            self.assertEqual(set(params), {'ids', 'z_0', 'x', 'z_0_feat_mask'})
            torch.testing.assert_close(params['z_0'][0], template, rtol=0, atol=0)
            calls.append(1)
            return raw.forward_tracking(**params)
        with torch.no_grad():
            for i in range(1, 71):
                frame = TrackerEvalData_FrameData(i, box if i != 30 else None, None, {'image': image})
                task = TrackerEvalData_TaskDesc(91, sequence if i == 1 else None, init if i == 1 else None, frame, i == 70)
                output = run_tracker_evaluator(evaluator, TrackerEvalData([task], {}), infer, raw)
                if i < 70:
                    self.assertEqual(len(raw.ltcp_memory_dicts[91]), min(i, 2))
        self.assertEqual(len(calls), 70)
        self.assertFalse(raw.ltcp_memory_dicts)
        self.assertFalse(pipelines[0].ctr._states)
        self.assertFalse(pipelines[0].all_tracking_task_local_contexts)
        result = output['evaluated_sequences'][0]
        self.assertEqual(result.output_box.shape, (71, 4))
        pred = result.output_box.copy(); pred[:, 2:] -= pred[:, :2]
        gt = result.groundtruth_box.copy(); gt[:, 2:] -= gt[:, :2]
        metrics = evaluate_sequence(pred, gt, result.groundtruth_object_existence_flag)
        self.assertEqual(int(metrics.valid.sum()), 70)
        evaluator.on_epoch_end()

    def test_float_frames_are_read_as_0_255_pixel_values(self):
        self.assertEqual(int(_to_gray_uint8(torch.ones(3, 8, 8)).max()), 1)


if __name__ == '__main__':
    unittest.main()
