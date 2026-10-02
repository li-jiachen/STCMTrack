"""Static checks for the SPMTrack baseline path: no model is built, no forward pass runs, torch is not imported.

Run: python -m unittest tests.test_spmtrack_static -v
(from the repository root; `python -m unittest discover -s tests` also picks it up)

They cover the pinned upstream provenance, the VARIANT -> model/component mapping, configuration values,
import isolation of the baseline from LTCP / CTR / STCMTrack, the Pn definition of the shared evaluation
core and the checkpoint guard of tools/check_spmtrack_weights.py (on tiny synthetic files). The training /
streaming behaviour of the SPMTrack classes needs a real forward pass and is not covered here.
"""
import ast
import hashlib
import json
from pathlib import Path
import re
import tempfile
import unittest

import numpy as np
from safetensors.numpy import save_file

from trackit.core.boot.funcs.utils.custom_yaml_loader import load_yaml
from trackit.core.evaluation.antiuav import evaluate_sequence
from tools import check_spmtrack_weights as weight_check
from tools import check_variant_mapping as mapping

ROOT = Path(__file__).resolve().parents[1]
PIPELINES = 'trackit/runner/evaluation/distributed/tracker_evaluator/default/pipelines'
SPMTRACK_SOURCES = (
    'trackit/models/methods/SPMTrack',
    f'{PIPELINES}/spmtrack_one_stream',
    'trackit/runner/evaluation/distributed/tracker_evaluator/components/post_process/spmtrack_box_with_score_map.py',
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def python_files(entry):
    path = ROOT / entry
    return [path] if path.is_file() else sorted(path.rglob('*.py'))


class UpstreamPinTests(unittest.TestCase):
    pin = json.loads((ROOT / 'trackit/models/methods/SPMTrack/UPSTREAM.json').read_text(encoding='utf-8'))

    def test_commit_and_license_are_pinned(self):
        self.assertEqual(self.pin['upstream_commit'], 'c581fe27231f3e16c38578e47daddadfaf6ffd7d')
        self.assertEqual(self.pin['upstream_repository'], 'https://github.com/WenRuiCai/SPMTrack')
        self.assertEqual(self.pin['license'], 'Apache-2.0')
        self.assertEqual(sha256(ROOT / 'LICENSE'), self.pin['license_sha256'])  # upstream and repo license are identical

    def test_verbatim_and_modified_files_match_the_record(self):
        for entry in self.pin['files']:
            with self.subTest(file=entry['local']):
                actual = sha256(ROOT / entry['local'])
                if entry['status'] == 'verbatim':
                    self.assertEqual(actual, entry['upstream_sha256'])
                elif entry['status'] == 'modified':
                    self.assertEqual(actual, entry['local_sha256'])
                    self.assertNotEqual(actual, entry['upstream_sha256'])
                    self.assertTrue(entry['deviation'])
                else:
                    self.assertEqual(entry['status'], 'ported')
                    self.assertTrue(entry['deviation'])  # every rewrite documents its deviations


class VariantMappingTests(unittest.TestCase):
    def test_mapping_is_consistent_for_both_datasets(self):
        for dataset in ('antiuav410', 'antiuav300'):
            with self.subTest(dataset=dataset):
                rows = mapping.compute(dataset)
                self.assertEqual(mapping.problems(rows), [])
                self.assertEqual(set(rows), {'baseline', 'spmtrack', 'stcm_base', 'ltcp', 'mcc', 'rgtc',
                                             'ltcp_mcc', 'ltcp_rgtc', 'mcc_rgtc', 'full'})

    def test_baseline_is_spmtrack_and_stcm_base_is_not(self):
        rows = mapping.compute('antiuav410')
        for name in ('baseline', 'spmtrack'):
            self.assertEqual(rows[name]['model']['classes'], ['SPMTrackInference_DINOv2', 'SPMTrack_DINOv2'])
            self.assertEqual((rows[name]['train_templates'], rows[name]['train_search_frames']), (3, 2))
        self.assertEqual(rows['stcm_base']['model']['classes'], ['STCMTrackInference_DINOv2', 'STCMTrack_DINOv2'])
        self.assertEqual(rows['stcm_base']['train_templates'], 1)

    def test_shared_mixins_are_identical_copies(self):
        for name in ('disable_torch_compile', 'evaluation', 'eval_short', 'dataset_antiuav300'):
            with self.subTest(mixin=name):
                self.assertEqual((ROOT / f'config/SPMTrack/_mixin/{name}.yaml').read_bytes(),
                                 (ROOT / f'config/STCMTrack/_mixin/{name}.yaml').read_bytes())


class SPMTrackConfigTests(unittest.TestCase):
    cfg = load_yaml(str(ROOT / 'config/SPMTrack/dinov2/config.yaml'))

    def test_model_and_structure(self):
        self.assertEqual(self.cfg['type'], 'SPMTrack')
        self.assertNotIn('ltcp', self.cfg['model'])
        self.assertFalse(self.cfg['model']['allow_unmarked_weights'])
        self.assertEqual(self.cfg['common']['template_size'], [196, 196])
        self.assertEqual(self.cfg['common']['search_region_size'], [378, 378])
        tmoe = self.cfg['model']['tmoe']
        self.assertEqual((tmoe['r'], tmoe['alpha'], tmoe['expert_nums'], tmoe['init_method']), (64, 64, 4, 'bert'))

    def test_training_follows_upstream_structure(self):
        train = self.cfg['run']['data']['train']
        positive = train['siamese_training_pair_sampling']['positive_sample']
        self.assertEqual((positive['sample_mode'], positive['num_template_frames'], positive['num_search_frames'],
                          positive['MAX_SAMPLE_INTERVAL']), ('interval', 3, 2, 400))
        flip, color, deit = train['transform']['augmentation']
        self.assertEqual(flip['target'], ['template_0', 'template_1', 'template_2', 'search_region_0', 'search_region_1'])
        self.assertFalse(flip['joint'])
        self.assertFalse(color['joint'])
        self.assertTrue(deit['joint'])
        self.assertNotIn('temporal_consistent_crops', train['transform'])
        criteria = self.cfg['run']['runner']['train']['criteria']
        self.assertEqual(criteria['frame_loss_reduction'], 'sum')
        self.assertTrue(criteria['classification']['iou_aware_classification_score'])

    def test_evaluation_follows_upstream_inference(self):
        pipeline = self.cfg['run']['runner']['test']['evaluator']['pipeline']
        self.assertEqual(pipeline['type'], 'spmtrack_one_stream_tracker')
        self.assertNotIn('ctr', pipeline)
        self.assertEqual(pipeline['search_region_cropping'], {'type': 'simple', 'min_object_size': 10, 'area_factor': 5.0})
        self.assertEqual(pipeline['post_process'], {'type': 'box_with_score_map_spmtrack', 'window_penalty': 0.45})
        self.assertEqual(pipeline['online_template']['area_factor'], 2.0)
        self.assertEqual([p['type'] for p in pipeline['plugin']], ['template_foreground_indicating_mask_generation'])

    def test_evaluation_uses_the_shared_metric_handlers(self):
        for task in ('test', 'eval'):
            handlers = self.cfg['run']['data'][task]['result_collector']['dispatch'][0]['handlers']
            self.assertEqual(handlers[0]['type'], 'one_pass_evaluation_compatible')

    def test_documented_protocol_adaptation(self):
        self.assertEqual(self.cfg['run']['num_epochs'], 80)
        self.assertEqual(self.cfg['run']['data']['train']['global_batch_size'], 4)
        self.assertFalse(self.cfg['run']['runner']['train']['torch_compile']['enabled'])
        self.assertFalse(self.cfg['run']['efficiency_assessment']['enabled'])

    def test_stcmtrack_config_keeps_its_own_definitions(self):
        stcm = load_yaml(str(ROOT / 'config/STCMTrack/dinov2/config.yaml'))
        pipeline = stcm['run']['runner']['test']['evaluator']['pipeline']
        self.assertEqual(stcm['type'], 'STCMTrack')
        self.assertEqual(pipeline['type'], 'one_stream_tracker')
        self.assertEqual(pipeline['post_process'], {'type': 'box_with_score_map', 'window_penalty': 0.0})
        positive = stcm['run']['data']['train']['siamese_training_pair_sampling']['positive_sample']
        self.assertEqual((positive['sample_mode'], positive['num_template_frames'], positive['num_search_frames']),
                         ('first_frame_causal', 1, 3))
        self.assertNotIn('frame_loss_reduction', stcm['run']['runner']['train']['criteria'])  # default 'mean'


class IsolationTests(unittest.TestCase):
    def test_baseline_sources_do_not_import_stcmtrack_ltcp_or_ctr(self):
        for entry in SPMTRACK_SOURCES:
            for path in python_files(entry):
                with self.subTest(file=str(path.relative_to(ROOT))):
                    for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                        modules = []
                        if isinstance(node, ast.Import):
                            modules = [alias.name for alias in node.names]
                        elif isinstance(node, ast.ImportFrom):
                            modules = [mapping.absolute_import_base(path, node)]
                            modules += [f'{modules[0]}.{alias.name}' for alias in node.names]
                        for module in modules:
                            parts = set(module.split('.'))
                            self.assertNotIn('STCMTrack', parts, module)
                            self.assertNotIn('ltcp', parts, module)
                            self.assertNotIn('ctr', parts, module)
                            self.assertNotIn('one_stream', parts, module)

    def test_baseline_text_does_not_instantiate_ltcp_or_ctr(self):
        for entry in SPMTRACK_SOURCES:
            for path in python_files(entry):
                code = re.sub(r'"""(.|\n)*?"""|#.*', '', path.read_text(encoding='utf-8'))
                with self.subTest(file=str(path.relative_to(ROOT))):
                    self.assertIsNone(re.search(r'LocalEnhancedTemporalContextPropagation|ConfidenceTriggeredRelocalization|'
                                                r'LTCPConfig|CTRConfig', code))


class PnDefinitionTests(unittest.TestCase):
    """Pn (paper Sec. 3.2): center error / ground-truth box diagonal, strictly below 0.5."""
    SHARED = 'trackit.core.evaluation.antiuav'

    def test_every_entry_point_calls_the_shared_core(self):
        entries = {
            'trackit/data/components/result_collector/handler/one_pass_evaluation/ope_metrics.py',
            'trackit/data/components/result_collector/handler/one_pass_evaluation_compatible/ope_metrics.py',
            'tools/evaluate_antiuav_iou_p20.py',
        }
        for entry in sorted(entries):
            with self.subTest(entry=entry):
                text = (ROOT / entry).read_text(encoding='utf-8')
                self.assertIn(f'from {self.SHARED} import', text)
                self.assertRegex(text, r'evaluate_sequence')

    def test_core_formula_text(self):
        core = (ROOT / 'trackit/core/evaluation/antiuav.py').read_text(encoding='utf-8')
        self.assertIn('np.hypot(g[:, 2], g[:, 3])', core)
        self.assertIn('(vn < .5).mean()', core)
        self.assertIn("'normalized_precision': 'center_error / hypot(gt_width, gt_height) < 0.5'", core)
        self.assertIn('Sec. 3.2', core)

    def test_boundary_values_on_synthetic_boxes(self):
        gt = [[0, 0, 30, 40]]  # diagonal 50
        pn = lambda dx: evaluate_sequence([[dx, 0, 30, 40]], gt).pn
        self.assertEqual(pn(24.9), 1.)   # 0.498 < 0.5
        self.assertEqual(pn(25.0), 0.)   # exactly 0.5 is not below 0.5
        result = evaluate_sequence([[20, 0, 30, 40]], gt)
        self.assertEqual((result.pn, result.p20), (1., 0.))  # 20 px: 0.4 of the diagonal, but not below 20 px

    def test_box_size_scale_is_the_gt_diagonal_not_the_prediction(self):
        gt = [[0, 0, 30, 40]]  # center (15, 20), diagonal 50
        tiny_prediction = evaluate_sequence([[20, 0, 3, 4]], gt)  # center (21.5, 2): offset (6.5, -18)
        self.assertAlmostEqual(tiny_prediction.normalized_error[0], np.hypot(6.5, 18.) / 50.)
        shifted = evaluate_sequence([[10, 0, 30, 40]], gt)  # same-size prediction, 10 px to the right
        self.assertAlmostEqual(shifted.normalized_error[0], 10. / 50.)


class WeightGuardTests(unittest.TestCase):
    @staticmethod
    def write(path, *, marker=True, scaling=True, alpha=64., foreign=None, drop=()):
        tensors = {'head.cls_mlp.layers.0.weight': np.zeros((2, 2), np.float32),
                   'track_query': np.zeros((1, 4), np.float32), 'query_embed': np.zeros((1, 4), np.float32),
                   'token_type_embed': np.zeros((3, 4), np.float32),
                   'blocks.0.attn.proj.tmoe.compress_expert': np.zeros((2, 4), np.float32)}
        if scaling:
            tensors['expert_alpha'] = np.array(alpha, np.float32)
            tensors['use_rsexpert'] = np.array(False)
        if marker:
            tensors['_spmtrack_port_version'] = np.array(1, np.int64)
        for key in foreign or ():
            tensors[key] = np.zeros((1,), np.float32)
        for key in drop:
            tensors.pop(key)
        save_file(tensors, str(path))

    def check(self, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'w.safetensors'
            self.write(path, **{k: v for k, v in kwargs.items() if k in ('marker', 'scaling', 'alpha', 'foreign', 'drop')})
            return weight_check.validate_spmtrack_weights(path, allow_unmarked=kwargs.get('allow_unmarked', False))

    def test_marked_file_is_accepted(self):
        self.assertEqual(self.check()['status'], 'passed')

    def test_unmarked_file_needs_an_explicit_declaration(self):
        with self.assertRaisesRegex(ValueError, 'marker'):
            self.check(marker=False)
        result = self.check(marker=False, allow_unmarked=True)
        self.assertIn('UNVERIFIED', result['provenance'])

    def test_stcmtrack_keys_are_always_rejected(self):
        for key in ('_paper_implementation_version', '_expert_alpha', '_use_rsexpert', 'ltcp.gate.weight'):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, 'STCMTrack'):
                    self.check(foreign=[key], allow_unmarked=True)

    def test_missing_or_mismatched_scaling_and_missing_parts_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'scaling'):
            self.check(scaling=False)
        with self.assertRaisesRegex(ValueError, 'alpha'):
            self.check(alpha=1.)
        for key in ('head.cls_mlp.layers.0.weight', 'track_query', 'blocks.0.attn.proj.tmoe.compress_expert'):
            with self.subTest(dropped=key), self.assertRaises(ValueError):
                self.check(drop=[key])


if __name__ == '__main__':
    unittest.main()
