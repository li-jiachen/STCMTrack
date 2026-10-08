"""Unit tests: model and checkpoints, LTCP (Sec. 2.2), CTR (Sec. 2.3), evaluation metrics (Sec. 3.2),
dataset cache and training-clip sampling.

Run: python -m unittest discover -s tests -v
The tests use the real modules with a small DINOv2 and synthetic images, and run on the CPU in a few seconds.
"""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import zipfile

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file, load_file
from trackit.models.backbone.dinov2 import DinoVisionTransformer
from trackit.models.methods.STCMTrack.STCMTrack import STCMTrack_DINOv2
from trackit.models.methods.STCMTrack.STCMTrack_inference import STCMTrackInference_DINOv2
from trackit.models.methods.STCMTrack.modules.ltcp import LTCPConfig, LocalEnhancedTemporalContextPropagation
from trackit.models import ModelManager, ModelBuildingContext, ModelImplSuggestions
from trackit.runner.evaluation.distributed.tracker_evaluator.default.pipelines.one_stream.ctr import (
    ConfidenceTriggeredRelocalization, CTRConfig, _TrackState)
from trackit.runner.evaluation.distributed.tracker_evaluator.components.post_process.box_with_score_map import PostProcessing_BoxWithScoreMap
from trackit.core.evaluation.antiuav import evaluate_sequence
from trackit.data.components.result_collector.handler.one_pass_evaluation_compatible.ope_metrics import (
    compute_one_pass_evaluation_metrics, DatasetOPEMetricsListBuilder, compute_OPE_metrics_mean)
from trackit.data.components.result_collector.handler.utils.compatibility import ExternalToolkitCompatibilityHelper
from trackit.data.components.result_collector.handler.external_adaptors.pytracking import (
    PyTrackingAnalysisModuleTrackingResultWriter, PyTrackingEvaluationToolAdaptor)
from trackit.data.methods.siamese_tracker_train.siamese_training_pair_sampling._algos import (
    sample_first_frame_causal, NoCausalClipError)
from tools import evaluate_antiuav_iou_p20 as external
from tools.export_stcmtrack_weights import export_weights
from tools.check_stcmtrack_weights import validate_weight_pair
from trackit.datasets.SOT.datasets._antiuav_layout import antiuav_cache_identity

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


def make_model(inference=False, stage2=True, ltcp=True):
    vit = DinoVisionTransformer(img_size=28, patch_size=14, embed_dim=32, depth=1,
                               num_heads=4, block_chunks=0, init_values=1e-5)
    cls = STCMTrackInference_DINOv2 if inference else STCMTrack_DINOv2
    return cls(vit, (1, 1), (2, 2), 4, 4., 0., expert_nums=2,
               ltcp_config={'enabled': ltcp, 'train_only': stage2, 'memory_dtype': 'float32', 'print_summary': False})


def inputs(batch=2):
    return (torch.randn(batch, 3, 14, 14), torch.ones(batch, 1, 1, dtype=torch.long),
            [torch.randn(batch, 3, 28, 28) for _ in range(3)])


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)

    def test_train_streaming_equivalence_and_history(self):
        train, infer = make_model().eval(), make_model(True).eval()
        infer.load_state_dict(train.state_dict())
        z, mask, xs = inputs()
        histories, lengths = [], []
        h1 = train.ltcp.register_forward_pre_hook(lambda _, args: histories.append(0 if args[1] is None else args[1].shape[1]))
        h2 = train.blocks[0].register_forward_pre_hook(lambda _, args: lengths.append(args[0].shape[1]))
        expected = train(z, mask, *xs)
        h1.remove(); h2.remove()
        self.assertEqual(histories, [0, 1, 2])
        self.assertEqual(lengths, [6, 6, 6])  # 1 query token + 1 template token + 4 search tokens
        for i, x in enumerate(xs):
            actual = infer.forward_tracking([8, 19], z, x, mask)
            for key in ('score_map', 'boxes'):
                torch.testing.assert_close(actual[key], expected[i][key], rtol=0, atol=0)
        with torch.no_grad():
            raw, _ = infer._encode_search(infer._z_feat(z, mask), xs[-1])
        torch.testing.assert_close(infer.ltcp_memory_dicts[8][-1], raw[0])
        infer.forget_tracking(8)
        self.assertNotIn(8, infer.ltcp_memory_dicts)
        self.assertIn(19, infer.ltcp_memory_dicts)
        infer.end_eval()
        self.assertFalse(infer.ltcp_memory_dicts)

    def test_stage2_optimizer_changes_only_gate(self):
        model = make_model().train()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        z, mask, xs = inputs()
        out = model(z, mask, *xs)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=.1)
        loss = sum(F.binary_cross_entropy_with_logits(o['score_map'], torch.ones_like(o['score_map'])) for o in out) / 3
        loss.backward(); optimizer.step()
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        self.assertEqual(trainable, ['ltcp.gate.weight', 'ltcp.gate.bias'])
        changed = {k for k, v in model.state_dict().items() if not torch.equal(v, before[k])}
        self.assertEqual(changed, set(trainable))
        self.assertTrue(torch.isfinite(loss))

    def test_queries_are_initialized_even_with_nan_fill(self):
        torch.use_deterministic_algorithms(True)
        try:
            model = make_model(stage2=False)
            for name in ('track_query', 'query_embed', 'token_type_embed'):
                self.assertTrue(torch.isfinite(getattr(model, name)).all())
        finally:
            torch.use_deterministic_algorithms(False)

    def test_manager_sync_preserves_frozen_network(self):
        manager = ModelManager(ModelBuildingContext(lambda advice: make_model(advice.optimize_for_inference),
                                                    lambda advice: 'infer' if advice.optimize_for_inference else 'train'))
        train = manager.create(torch.device('cpu'), ModelImplSuggestions())
        with torch.no_grad():
            train.model.track_query.fill_(.125)
            train.model.head.cls_mlp.layers[0].weight.fill_(.234)
        train.notify_update()
        with contextlib.redirect_stdout(io.StringIO()):
            infer = manager.create(torch.device('cpu'), ModelImplSuggestions(optimize_for_inference=True))
        for key, value in train.model.state_dict().items():
            torch.testing.assert_close(value, infer.model.state_dict()[key], rtol=0, atol=0)

    def test_checkpoint_export_and_loading(self):
        model = make_model()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            save_file(model.state_dict(), str(tmp / 'model.bin'))
            export_weights(tmp / 'model.bin', tmp / 'base.safetensors', tmp / 'ltcp.safetensors')
            loaded = make_model(True)
            loaded.load_state_dict(load_file(str(tmp / 'base.safetensors')), strict=False)
            loaded.load_state_dict(load_file(str(tmp / 'ltcp.safetensors')), strict=False)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, loaded.state_dict()[key], rtol=0, atol=0)
            self.assertTrue((tmp / 'base.manifest.json').is_file())
        with self.assertRaisesRegex(ValueError, 'before the LTCP'):
            make_model(True).load_state_dict(model.export_ltcp_state_dict(), strict=False)
        without_scaling = {k: v for k, v in model.state_dict().items() if not k.startswith('_')}
        with self.assertRaisesRegex(ValueError, 'Not an STCMTrack checkpoint'):
            model.load_state_dict(without_scaling, strict=False)
        missing = dict(model.state_dict()); missing.pop('track_query')
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            model.load_state_dict(missing, strict=False)

    def test_additional_metadata_entries_are_ignored(self):
        model = make_model()
        extra = {'_note': torch.tensor(1, dtype=torch.int64)}
        base = {k: v for k, v in model.state_dict().items() if not k.startswith('ltcp.')}
        full, split = make_model(True), make_model(True)
        result = full.load_state_dict({**model.state_dict(), **extra}, strict=True)
        self.assertFalse(result.missing_keys or result.unexpected_keys)
        split.load_state_dict({**base, **extra}, strict=False)
        result = split.load_state_dict({**model.export_ltcp_state_dict(), **extra}, strict=False)
        self.assertFalse(result.unexpected_keys)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, full.state_dict()[key], rtol=0, atol=0)
            torch.testing.assert_close(value, split.state_dict()[key], rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            save_file({**model.state_dict(), **extra}, str(tmp / 'model.bin'))
            export_weights(tmp / 'model.bin', tmp / 'base.safetensors', tmp / 'ltcp.safetensors')
            self.assertEqual(set(load_file(str(tmp / 'base.safetensors'))), set(base))
            self.assertEqual(validate_weight_pair(tmp / 'base.safetensors', tmp / 'ltcp.safetensors')['status'], 'passed')
            provenance = {'source_checkpoint_sha256': 'same'}
            save_file({**base, **extra}, str(tmp / 'b.safetensors'), metadata={**provenance, 'component': 'base'})
            save_file({**model.export_ltcp_state_dict(), **extra}, str(tmp / 'l.safetensors'),
                      metadata={**provenance, 'component': 'ltcp'})
            self.assertEqual(validate_weight_pair(tmp / 'b.safetensors', tmp / 'l.safetensors')['status'], 'passed')

    def test_stage1_checkpoint_initializes_stage2(self):
        base, stage2 = make_model(stage2=False, ltcp=False), make_model()
        result = stage2.load_state_dict(base.state_dict(), strict=False)
        self.assertEqual(set(result.missing_keys), {'ltcp.gate.weight', 'ltcp.gate.bias'})
        self.assertFalse(result.unexpected_keys)

    def test_partial_ltcp_gate_is_rejected(self):
        model = make_model()
        for key in ('ltcp.gate.weight', 'ltcp.gate.bias'):
            state = dict(model.state_dict()); state.pop(key)
            with self.assertRaisesRegex(ValueError, 'Incomplete.*LTCP'):
                make_model(True).load_state_dict(state, strict=False)

    def test_export_pair_provenance_and_path_collision(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for name in ('first', 'second'):
                save_file(make_model().state_dict(), str(tmp / f'{name}.bin'))
                export_weights(tmp / f'{name}.bin', tmp / f'{name}.safetensors', tmp / f'{name}_ltcp.safetensors')
            result = validate_weight_pair(tmp / 'first.safetensors', tmp / 'first_ltcp.safetensors')
            self.assertEqual(result['status'], 'passed')
            self.assertEqual(validate_weight_pair(tmp / 'first.safetensors')['status'], 'passed')
            with self.assertRaisesRegex(ValueError, 'provenance'):
                validate_weight_pair(tmp / 'first.safetensors', tmp / 'second_ltcp.safetensors')
            with self.assertRaisesRegex(ValueError, 'distinct'):
                export_weights(tmp / 'first.bin', tmp / 'third.safetensors', tmp / 'third.manifest.json')
            self.assertFalse((tmp / 'third.safetensors').exists())

    def test_ltcp_equations_1_to_3(self):
        x, mem, q = torch.randn(2, 5, 4), torch.randn(2, 2, 5, 4), torch.randn(2, 1, 4)
        ltcp = LocalEnhancedTemporalContextPropagation(4, LTCPConfig(enabled=True))
        with torch.no_grad():
            ltcp.gate.weight.copy_(torch.tensor([[1.2, -.1, .3, -.2, .4]])); ltcp.gate.bias.fill_(-.7)
        local = ((F.normalize(x, dim=-1).unsqueeze(1) * F.normalize(mem, dim=-1)).sum(-1).max(1).values + 1) / 2
        sim = (F.normalize(x.mean(1), dim=-1).unsqueeze(1) * F.normalize(mem.mean(2), dim=-1)).sum(-1)
        weight = sim.softmax(1)
        global_conf = ((weight * sim).sum(1, keepdim=True) + 1) / 2
        context = (weight[:, :, None, None] * mem).sum(1)
        gate = torch.sigmoid(F.linear(torch.cat([(local * global_conf).unsqueeze(-1), q.expand(-1, 5, -1)], -1),
                                      ltcp.gate.weight, ltcp.gate.bias)) * .05
        torch.testing.assert_close(ltcp(x, mem, q), (1 - gate) * x + gate * context)
        torch.testing.assert_close(ltcp(x, None, q), x, rtol=0, atol=0)


class CTRTests(unittest.TestCase):
    def setUp(self):
        self.ctr = ConfidenceTriggeredRelocalization(CTRConfig(enabled=True, print_summary=False))

    def test_residual_threshold_is_median_plus_alpha_mad(self):
        previous = np.zeros((10, 10), np.uint8)
        state = _TrackState(previous, np.array([2., 2., 4., 4.]), cv2.createBackgroundSubtractorMOG2(), homography=np.eye(3))
        current = previous.copy(); current[5, 5] = 5
        self.assertEqual(int((self.ctr._residual_mask(state, current) > 0).sum()), 1)
        current[:] = 100; current[5, 5] = 101
        self.assertEqual(int((self.ctr._residual_mask(state, current) > 0).sum()), 1)
        current = np.arange(100, dtype=np.uint8).reshape(10, 10)
        med = np.median(current); threshold = med + 4.4478 * np.median(abs(current.astype(float) - med))
        np.testing.assert_array_equal(self.ctr._residual_mask(state, current) > 0, current > threshold)

    def test_residual_valid_region_excludes_warp_padding(self):
        previous = np.zeros((10, 10), np.uint8)
        state = _TrackState(previous, None, None, homography=np.array([[1., 0., 2.], [0., 1., 0.], [0., 0., 1.]]))
        current = previous.copy(); current[:, :2] = 255; current[5, 5] = 1
        mask = self.ctr._residual_mask(state, current)
        self.assertEqual(int((mask > 0).sum()), 1)
        self.assertFalse(mask[:, :2].any())

    def test_foreground_mask_is_union_of_mog2_and_residual(self):
        residual, foreground = np.zeros((8, 8), np.uint8), np.zeros((8, 8), np.uint8)
        residual[2, 2] = 255; foreground[5, 5] = 255
        state = _TrackState(None, None, None)
        with patch.object(self.ctr, '_residual_mask', return_value=residual), patch.object(self.ctr, '_mog2_mask', return_value=foreground):
            np.testing.assert_array_equal(self.ctr._foreground_mask(state, foreground), residual | foreground)

    def test_mcc_keeps_previous_size_at_border(self):
        H = np.array([[1., 0., 8.], [0., 1., 0.], [0., 0., 1.]])
        bbox = self.ctr._map_previous_box(H, np.array([85., 40., 95., 60.]), np.array([100, 100]))
        np.testing.assert_allclose(bbox, [93., 40., 103., 60.])

    def test_homography_is_estimated_once_per_frame_and_reused_by_rgtc(self):
        img = torch.zeros(3, 20, 20)
        self.ctr.reset(3, img, np.array([2., 2., 4., 4.]))
        H = np.eye(3)
        with patch.object(self.ctr, '_estimate_homography', return_value=H) as estimate, patch.object(self.ctr, '_map_previous_box', return_value=None):
            self.assertIsNone(self.ctr.compensate_search_bbox(3, img, np.array([20, 20])))
            self.ctr.correct_prediction(3, img, np.array([2., 2., 4., 4.]), .1, np.array([20, 20]))
            self.assertEqual(estimate.call_count, 1)
        np.testing.assert_array_equal(self.ctr._states[3].homography, H)

    def test_nearest_candidate_tau_and_no_candidate_fallback(self):
        prior = np.array([100., 100., 110., 110.]); near = np.array([95., 95., 115., 115.]); far = prior + [200, 0, 200, 0]
        selected, _ = self.ctr._select_candidate([far, near], prior, prior, np.array([640, 480]))
        np.testing.assert_array_equal(selected, near)
        image = torch.zeros(3, 30, 30); box = np.array([10., 10., 20., 20.]); size = np.array([30, 30])
        self.ctr.reset(2, image, box)
        with patch.object(self.ctr, '_foreground_mask', return_value=np.zeros((30, 30), np.uint8)) as foreground:
            returned, corrected = self.ctr.correct_prediction(2, image, box, .4, size)
            foreground.assert_not_called(); self.assertFalse(corrected)
            self.ctr.correct_prediction(2, image, box, .399, size)
            self.assertEqual(foreground.call_count, 1)
            np.testing.assert_array_equal(returned, box)

    def test_confidence_is_sampled_at_regressed_center(self):
        post = PostProcessing_BoxWithScoreMap(torch.device('cpu'), (3, 3), (30, 30))
        post.start()
        probs = torch.tensor([[[.9, .1, .1], [.1, .3, .1], [.1, .1, .1]]])
        boxes = torch.tensor([0., 0., 1., 1.]).expand(1, 3, 3, 4).clone()
        result = post({'score_map': torch.logit(probs), 'boxes': boxes})
        torch.testing.assert_close(result['confidence'], torch.tensor([.3]))
        post.stop()


class MetricsTests(unittest.TestCase):
    def test_diagonal_thresholds_exact_auc_and_absence(self):
        result = evaluate_sequence([[20, 0, 30, 40], [25, 0, 30, 40]], [[0, 0, 30, 40]] * 2)
        self.assertEqual(result.p20, 0.); self.assertEqual(result.pn, .5)
        self.assertAlmostEqual(result.auc, (0.2 + 5 / 55) / 2)
        self.assertEqual(evaluate_sequence([[100, 100, 10, 10]], [[0, 0, 10, 10]]).auc, 0.)
        absent = evaluate_sequence([[0, 0, 10, 10], [0, 0, 0, 0]], [[0, 0, 10, 10], [np.nan]*4], [1, 0])
        self.assertEqual((absent.auc, absent.p20, absent.pn), (1., 1., 1.))
        invalid = evaluate_sequence([[0, 0, 0, 0]], [[0, 0, 10, 10]])
        self.assertEqual((invalid.auc, invalid.p20, invalid.pn), (0., 0., 0.))

    def test_internal_exported_metrics_and_dataset_aggregation_agree(self):
        gt = np.array([[0., 0., 30., 40.], [1., 1., 31., 41.], [0., 0., 0., 0.]])  # XYXY
        pred = np.array([[0., 0., 30., 40.], [21., 1., 51., 41.], [10., 10., 30., 30.]])
        flags = [1, 1, 0]
        internal, _ = compute_one_pass_evaluation_metrics('ANTIUAV410', pred, gt, flags, np.ones(3), ExternalToolkitCompatibilityHelper())
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); (tmp / 'gt' / 's').mkdir(parents=True)
            gt_xywh = gt.copy(); gt_xywh[:, 2:] -= gt_xywh[:, :2]
            pred_xywh = pred.copy(); pred_xywh[:, 2:] -= pred_xywh[:, :2]
            (tmp / 'gt/s/IR_label.json').write_text(json.dumps({'gt_rect': gt_xywh.tolist(), 'exist': flags}))
            writer = PyTrackingAnalysisModuleTrackingResultWriter(str(tmp), 'results')
            writer.write('tracker', None, 's', pred_xywh); writer.close()
            exported, _ = external.evaluate_antiuav_results(tmp / 'results.zip', tmp / 'gt')
            self.assertEqual((internal.auc, internal.precision_score, internal.normalized_precision_score),
                             (exported.auc, exported.precision_at_20, exported.norm_precision_at_05))
        from trackit.data.components.result_collector.handler.one_pass_evaluation.ope_metrics import compute_one_pass_evaluation_metrics as alternate_ope
        alternate, _ = alternate_ope(pred, gt, flags, np.ones(3))
        self.assertEqual((alternate.success_score, alternate.precision_score, alternate.normalized_precision_score),
                         (internal.auc, internal.precision_score, internal.normalized_precision_score))
        builder = DatasetOPEMetricsListBuilder(); builder.append('s', internal); builder.append('a', internal)
        dataset = builder.build().sort_by_sequence_name()
        self.assertEqual(dataset.get_mean().auc, internal.auc)
        self.assertEqual(dataset.get_success_score().tolist(), [internal.auc]*2)
        self.assertEqual(dataset.get_normalized_precision_score().tolist(), [internal.normalized_precision_score]*2)
        self.assertEqual(compute_OPE_metrics_mean([internal, internal]).auc, internal.auc)

    def test_exported_result_files_are_named_after_the_sequences(self):
        pred = np.array([[0., 0., 30., 40.], [21., 1., 51., 41.]])  # XYXY, as produced by the tracking pipeline
        labels = {'gt_rect': [[0, 0, 30, 40], [1, 1, 30, 40]], 'exist': [1, 1]}
        for dataset in ('ANTIUAV410', 'AntiUAV300'):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp); (tmp / 'gt' / 'seq_01').mkdir(parents=True)
                (tmp / 'gt/seq_01/IR_label.json').write_text(json.dumps(labels))
                handler = PyTrackingEvaluationToolAdaptor('tracker', str(tmp), 'results', False)
                handler.accept([SimpleNamespace(sequence_info=SimpleNamespace(dataset_name=dataset, sequence_name='seq_01'),
                                                output_box=pred)],
                               [SimpleNamespace(repeat_index=0, this_dataset=SimpleNamespace(total_repeat_times=1))])
                handler.close()
                with zipfile.ZipFile(tmp / 'results.zip') as archive:
                    self.assertEqual(archive.namelist(), ['tracker/seq_01.txt'])
                metric, rows = external.evaluate_antiuav_results(tmp / 'results.zip', tmp / 'gt')
                self.assertEqual((metric.matched_sequences, rows[0].seq_name, rows[0].used_frames), (1, 'seq_01', 2))
                self.assertAlmostEqual(metric.auc, (1. + 400 / 2000) / 2)  # IoU 1 and, for the 20-pixel shift, 0.2
                self.assertEqual(metric.precision_at_20, .5)               # an error of exactly 20 pixels is not below 20

    def test_malformed_or_partial_predictions_fail_explicitly(self):
        for text in ('0 0 10 10\nbad line\n0 0 10 10\n', 'nan 0 10 10\n', '0 0 10 10\n\n', ''):
            with self.assertRaises(ValueError):
                external.parse_prediction_text(text)
        with self.assertRaises(ValueError):
            external.evaluate_sequence('s', 's', [[0, 0, 10, 10]], [[0, 0, 10, 10]]*2)
        partial = external.evaluate_sequence('s', 's', [[0, 0, 10, 10]], [[0, 0, 10, 10]]*2, allow_partial=True)
        self.assertEqual(partial.used_frames, 1)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for name in ('a', 'b'):
                (tmp / name).mkdir(); (tmp / name / 'IR_label.json').write_text(json.dumps({'gt_rect': [[0, 0, 10, 10]], 'exist': [1]}))
            with zipfile.ZipFile(tmp / 'r.zip', 'w') as z:
                z.writestr('a.txt', '0 0 10 10\n')
            with self.assertRaisesRegex(ValueError, 'missing'):
                external.evaluate_antiuav_results(tmp / 'r.zip', tmp)
            with self.assertRaisesRegex(ValueError, 'expected'):
                external.evaluate_antiuav_results(tmp / 'r.zip', tmp, allow_partial=True, expected_sequences=120)
            metric, _ = external.evaluate_antiuav_results(tmp / 'r.zip', tmp, allow_partial=True)
            self.assertEqual(metric.skipped, 1)
            with zipfile.ZipFile(tmp / 'r.zip', 'w') as z:
                z.writestr('one/a.txt', '0 0 10 10\n'); z.writestr('two/a.txt', '0 0 10 10\n')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                external.evaluate_antiuav_results(tmp / 'r.zip', tmp, allow_partial=True)

    def test_missing_ground_truth_label_is_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); (tmp / 'gt' / 'a').mkdir(parents=True)
            with zipfile.ZipFile(tmp / 'r.zip', 'w') as archive:
                archive.writestr('a.txt', '0 0 10 10\n')
            with self.assertRaisesRegex(ValueError, 'IR_label.json'):
                external.evaluate_antiuav_results(tmp / 'r.zip', tmp / 'gt', allow_partial=True)


class DatasetCacheTests(unittest.TestCase):
    def test_root_labels_and_sequence_changes_invalidate_cache(self):
        from trackit.datasets.SOT.datasets.ANTIUAV import ANTIUAV_Seed
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ('first', 'second'):
                (root / split / 'a').mkdir(parents=True)
                (root / split / 'a' / 'IR_label.json').write_text('{"exist":[1],"gt_rect":[[0,0,10,10]]}')
            first = antiuav_cache_identity(root / 'first')
            self.assertNotEqual(first, antiuav_cache_identity(root / 'second'))
            self.assertEqual(ANTIUAV_Seed(str(root / 'first')).cache_identity, first)
            (root / 'first/a/IR_label.json').write_text('{"exist":[1],"gt_rect":[[1,0,10,10]]}')
            changed = antiuav_cache_identity(root / 'first')
            self.assertNotEqual(first, changed)
            (root / 'first/b').mkdir()
            with self.assertRaises(FileNotFoundError):
                antiuav_cache_identity(root / 'first')
            (root / 'first/b/IR_label.json').write_text('{"exist":[1],"gt_rect":[[0,0,10,10]]}')
            self.assertNotEqual(changed, antiuav_cache_identity(root / 'first'))


class SamplingTests(unittest.TestCase):
    def test_first_template_and_contiguous_causal_clip(self):
        rng = np.random.default_rng(11); valid = np.ones(20, bool); valid[[5, 9]] = False
        for _ in range(50):
            z, x = sample_first_frame_causal(20, valid, 1, 3, rng)
            self.assertEqual(z, [0]); self.assertGreater(x[0], 0)
            self.assertEqual(x, list(range(x[0], x[0]+3))); self.assertTrue(valid[x].all())
        with self.assertRaises(NoCausalClipError):
            sample_first_frame_causal(5, [1, 1, 0, 1, 1], 1, 3, rng)
        with self.assertRaises(NoCausalClipError):
            sample_first_frame_causal(5, [0, 1, 1, 1, 1], 1, 3, rng)


if __name__ == '__main__':
    unittest.main()
