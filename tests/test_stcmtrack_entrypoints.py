from pathlib import Path
import re
import unittest

import numpy as np

from trackit.core.boot.funcs.mixin import apply_static_mixin_rules
from trackit.core.boot.funcs.utils.custom_yaml_loader import load_yaml
from trackit.core.evaluation.antiuav import evaluate_sequence


ROOT = Path(__file__).resolve().parents[1]
DATASET_SPLICE = '${dataset_mixins[@]+"${dataset_mixins[@]}"}'


def dataset_mixins(script, dataset):
    match = re.search(r'case "\$DATASET" in\n(.*?)\nesac', script, re.S)
    if match is None:
        raise ValueError('Dataset selection block is missing')
    branch = re.search(r'^\s*' + re.escape(dataset) + r'\)(.*?);;', match.group(1), re.S | re.M)
    if branch is None:
        raise ValueError(f'Dataset {dataset} is not supported')
    assignment = re.search(r'dataset_mixins=\(([^)]*)\)', branch.group(1))
    if assignment is None:
        raise ValueError(f'Dataset {dataset} has no mixin selection')
    return assignment.group(1).split()


def initial_mixins(script, dataset):
    match = re.search(r'^mixin_names=\(([^\n]*)\)$', script, re.M)
    if match is None:
        raise ValueError('Mixin initialization is missing')
    tokens = match.group(1).split()
    if tokens.count(DATASET_SPLICE) != 1:
        raise ValueError('Dataset mixins must be applied exactly once')
    result = []
    for token in tokens:
        result.extend(dataset_mixins(script, dataset) if token == DATASET_SPLICE else [token])
    if any(re.fullmatch(r'\w+', token) is None for token in result):
        raise ValueError('Unexpected dynamic mixin selection')
    return result


def stage_mixins(script, dataset, stage):
    match = re.search(r'^if \[\[ "\$TRAIN_STAGE" == 1 \]\]; then\n(.*?)\nelse\n(.*?)\nfi',
                      script, re.S | re.M)
    if match is None:
        raise ValueError('Two-stage training selection is missing')
    result = initial_mixins(script, dataset)
    for assignment in re.findall(r'^\s*mixin_names\+=\(([^)]*)\)', match.group(stage), re.M):
        result.extend(assignment.split())
    return result


def build_config(mixins):
    cfg = load_yaml(str(ROOT / 'config/STCMTrack/dinov2/config.yaml'))
    for name in mixins:
        apply_static_mixin_rules(load_yaml(str(ROOT / f'config/STCMTrack/_mixin/{name}.yaml')), cfg)
    return cfg


class FullModelEntryTests(unittest.TestCase):
    def test_evaluation_entry_builds_full_model_for_both_datasets(self):
        script = (ROOT / 'test_stcmtrack.sh').read_text(encoding='utf-8')
        self.assertRegex(script, re.compile(r'^method_name=STCMTrack$', re.M))
        self.assertIn('"$REPO_ROOT/boot.sh" "$method_name" dinov2', script)
        self.assertNotRegex(script, r'(?m)^\s*(baseline|ltcp|mcc|rgtc|ltcp_mcc|ltcp_rgtc|mcc_rgtc)\)')
        for dataset, expected_name in (('antiuav410', 'ANTIUAV'), ('antiuav300', 'AntiUAV300')):
            with self.subTest(dataset=dataset):
                mixins = initial_mixins(script, dataset)
                cfg = build_config(mixins)
                self.assertEqual(cfg['type'], 'STCMTrack')
                self.assertTrue(cfg['model']['ltcp']['enabled'])
                self.assertEqual(cfg['model']['ltcp']['memory_size'], 2)
                pipeline = cfg['run']['runner']['test']['evaluator']['pipeline']
                self.assertEqual(pipeline['type'], 'one_stream_tracker')
                ctr = pipeline['ctr']
                self.assertTrue(ctr['enabled'] and ctr['mcc_enabled'] and ctr['rgtc_enabled'])
                self.assertEqual((ctr['confidence_threshold'], ctr['ransac_reproj_threshold']), (.40, 2.0))
                self.assertEqual(set(cfg['run']['data']), {'eval'})
                source = cfg['run']['data']['eval']['source']['parameters']['datasets']
                self.assertEqual(source, [{'name': expected_name, 'type': 'SOT', 'splits': ['test']}])
                self.assertTrue(cfg['run']['data']['eval']['transform']['with_full_template_image'])

    def test_full_evaluation_checks_both_weight_components_before_cuda(self):
        script = (ROOT / 'test_stcmtrack.sh').read_text(encoding='utf-8')
        preflight = 'python3 "$REPO_ROOT/tools/check_stcmtrack_weights.py"'
        self.assertIn(preflight, script)
        self.assertIn('--base "$BASE_WEIGHT"', script)
        self.assertIn('--ltcp "$LTCP_WEIGHT"', script)
        self.assertIn('--weight_path "$BASE_WEIGHT"', script)
        self.assertIn('--weight_path "$LTCP_WEIGHT"', script)
        self.assertNotIn('use_ltcp=false', script)
        self.assertLess(script.index(preflight), script.index('python3 - "$first_device_id"'))


class TrainingEntryTests(unittest.TestCase):
    def test_entry_builds_both_training_stages_with_matching_datasets(self):
        script = (ROOT / 'train_stcmtrack.sh').read_text(encoding='utf-8')
        self.assertIn('"$REPO_ROOT/boot.sh" STCMTrack dinov2', script)
        expected_sources = {
            'antiuav410': ('ANTIUAV_Train', 'ANTIUAV_Val'),
            'antiuav300': ('AntiUAV300_Train', 'AntiUAV300_Val'),
        }
        for dataset, names in expected_sources.items():
            for stage in (1, 2):
                with self.subTest(dataset=dataset, stage=stage):
                    cfg = build_config(stage_mixins(script, dataset, stage))
                    self.assertEqual(cfg['run']['num_epochs'], 80 if stage == 1 else 20)
                    ltcp = cfg['model'].get('ltcp', {})
                    self.assertEqual(bool(ltcp.get('enabled')), stage == 2)
                    self.assertEqual(bool(ltcp.get('train_only')), stage == 2)
                    for task, name in zip(('train', 'val'), names):
                        source = cfg['run']['data'][task]['source']['parameters']['datasets']
                        self.assertEqual(source, [{'name': name, 'type': 'SOT', 'splits': [task]}])
                    optimizer = cfg['run']['runner']['train']['optimization']['optimizer']
                    self.assertEqual((optimizer['type'], optimizer['lr'], optimizer['weight_decay']),
                                     ('AdamW', 1.e-4, .1))

    def test_stage2_requires_and_loads_the_checked_stage1_checkpoint(self):
        script = (ROOT / 'train_stcmtrack.sh').read_text(encoding='utf-8')
        branches = re.search(r'^if \[\[ "\$TRAIN_STAGE" == 1 \]\]; then\n(.*?)\nelse\n(.*?)\nfi',
                             script, re.S | re.M)
        self.assertIsNotNone(branches)
        stage1, stage2 = branches.groups()
        self.assertNotIn('--weight_path', stage1)
        self.assertRegex(stage2, re.compile(
            r'if \[\[ -z "\$BASE_WEIGHT" \|\| ! -f "\$BASE_WEIGHT" \]\]; then.*?exit 1\s*fi', re.S))
        self.assertIn('weight_args+=(--weight_path "$BASE_WEIGHT")', stage2)
        self.assertIn('${weight_args[@]+"${weight_args[@]}"}', script)
        preflight = 'python3 "$REPO_ROOT/tools/check_stcmtrack_weights.py" --base "$BASE_WEIGHT"'
        self.assertIn(preflight, stage2)
        self.assertLess(script.index(preflight), script.index('python3 - "$first_device_id"'))


class PnDefinitionTests(unittest.TestCase):
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

    def test_boundary_values_on_synthetic_boxes(self):
        gt = [[0, 0, 30, 40]]
        pn = lambda dx: evaluate_sequence([[dx, 0, 30, 40]], gt).pn
        self.assertEqual(pn(24.9), 1.)
        self.assertEqual(pn(25.0), 0.)
        result = evaluate_sequence([[20, 0, 30, 40]], gt)
        self.assertEqual((result.pn, result.p20), (1., 0.))

    def test_box_size_scale_is_the_gt_diagonal_not_the_prediction(self):
        gt = [[0, 0, 30, 40]]
        tiny_prediction = evaluate_sequence([[20, 0, 3, 4]], gt)
        self.assertAlmostEqual(tiny_prediction.normalized_error[0], np.hypot(6.5, 18.) / 50.)
        shifted = evaluate_sequence([[10, 0, 30, 40]], gt)
        self.assertAlmostEqual(shifted.normalized_error[0], 10. / 50.)


if __name__ == '__main__':
    unittest.main()
