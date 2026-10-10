
import ast
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

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
                    self.assertTrue(entry['deviation'])  


class VariantMappingTests(unittest.TestCase):
    def test_mapping_is_consistent_for_both_datasets(self):
        for dataset in ('antiuav410', 'antiuav300'):
            with self.subTest(dataset=dataset):
                rows = mapping.compute(dataset)
                self.assertEqual(mapping.problems(rows), [])
                self.assertEqual(set(rows), {'baseline', 'ltcp', 'mcc', 'rgtc',
                                             'ltcp_mcc', 'ltcp_rgtc', 'mcc_rgtc', 'full'})

    def test_table2_row1_is_the_same_stcmtrack_with_all_components_off(self):
        for dataset in ('antiuav410', 'antiuav300'):
            rows = mapping.compute(dataset)
            baseline = rows['baseline']
            with self.subTest(dataset=dataset):
                self.assertEqual(set(mapping.STCM_VARIANTS), set(mapping.COMPONENTS))
                self.assertEqual(baseline['method'], 'STCMTrack')
                self.assertEqual((baseline['ltcp'], baseline['mcc'], baseline['rgtc']), (False, False, False))
                self.assertEqual(mapping.ablation_conflicts(rows), [])
            for name, row in rows.items():
                with self.subTest(dataset=dataset, variant=name):
                    self.assertEqual(row['model']['classes'], ['STCMTrackInference_DINOv2', 'STCMTrack_DINOv2'])
                    self.assertEqual((row['train_templates'], row['train_search_frames'], row['train_sample_mode']),
                                     (1, 3, 'first_frame_causal'))
                    self.assertEqual(row['pipeline_type'], 'one_stream_tracker')
                    self.assertEqual(row['window_penalty'], 0.)
                    self.assertEqual(row['model'], baseline['model'])
                    self.assertEqual(row['pipeline'], baseline['pipeline'])
                    for key in ('shared_settings_sha256', 'eval_shared_settings_sha256'):
                        self.assertEqual(row[key], baseline[key])

    def test_check_rejects_extra_variants_and_an_independent_spm_control(self):
        rows = mapping.compute('antiuav410')
        rows['stcm_base'] = copy.deepcopy(rows['ltcp'])
        self.assertTrue(any('exactly the eight' in problem for problem in mapping.problems(rows)))
        rows = mapping.compute('antiuav410')
        rows['baseline'].update(method='SPMTrack', config_type='SPMTrack',
                                model=mapping.resolve_model('SPMTrack'),
                                pipeline=mapping.resolve_pipeline('spmtrack_one_stream_tracker'))
        self.assertTrue(any('all eight Table 2 variants must select STCMTrack' in problem
                            for problem in mapping.problems(rows)))

    def test_check_rejects_every_baseline_component_being_enabled(self):
        original_rows = mapping.compute('antiuav410')
        for component in ('ltcp', 'mcc', 'rgtc'):
            with self.subTest(component=component):
                rows = copy.deepcopy(original_rows)
                rows['baseline'][component] = True
                self.assertTrue(any('baseline: components' in problem for problem in mapping.problems(rows)))

    @staticmethod
    def ablation_config(name, dataset='antiuav410', evaluation=False):
        variants, datasets = mapping.parse_variants(mapping.SCRIPT.read_text(encoding='utf-8'))
        info = variants[name]
        mixins = ['disable_torch_compile'] + datasets[dataset] + info['variant_mixins']
        if evaluation:
            mixins.append('evaluation')
        return mapping.build_config(info['method'], mixins)

    def test_all_eight_stcm_rows_share_the_complete_non_component_config(self):
        for dataset in ('antiuav410', 'antiuav300'):
            for evaluation in (False, True):
                reference = mapping.shared_config(self.ablation_config('ltcp', dataset, evaluation))
                for name in mapping.STCM_VARIANTS:
                    with self.subTest(dataset=dataset, evaluation=evaluation, variant=name):
                        config = self.ablation_config(name, dataset, evaluation)
                        self.assertEqual(mapping.shared_config(config), reference)

    def test_baseline_template_sampling_pipeline_and_hash_drift_are_failures(self):
        changes = (
            ('templates', 'train_templates', 3),
            ('search_frames', 'train_search_frames', 2),
            ('sampling', 'train_sample_mode', 'interval'),
            ('pipeline', 'pipeline_type', 'spmtrack_one_stream_tracker'),
            ('train_hash', 'shared_settings_sha256', 'different'),
            ('eval_hash', 'eval_shared_settings_sha256', 'different'),
        )
        for dataset in ('antiuav410', 'antiuav300'):
            original_rows = mapping.compute(dataset)
            for label, key, value in changes:
                with self.subTest(dataset=dataset, setting=label):
                    rows = copy.deepcopy(original_rows)
                    rows['baseline'][key] = value
                    conflicts = mapping.ablation_conflicts(rows)
                    self.assertTrue(any('baseline: non-component settings differ' in issue and key in issue
                                        for issue in conflicts))
                    self.assertTrue(set(conflicts).issubset(mapping.problems(rows)))

    def test_real_config_changes_to_the_control_are_detected_after_rebuilding(self):
        changes = (
            ('templates', lambda cfg: cfg['run']['data']['train']['siamese_training_pair_sampling'][
                'positive_sample'].update(num_template_frames=3)),
            ('sampling', lambda cfg: cfg['run']['data']['train']['siamese_training_pair_sampling'][
                'positive_sample'].update(sample_mode='interval')),
            ('pipeline', lambda cfg: cfg['run']['runner']['test']['evaluator']['pipeline'].update(
                type='spmtrack_one_stream_tracker')),
            ('gradient_clip', lambda cfg: cfg['run']['runner']['train']['optimization'].update(max_grad_norm=2.)),
        )
        original_builder = mapping.build_config
        for dataset in ('antiuav410', 'antiuav300'):
            for label, change in changes:
                def changed_control(method, mixins):
                    config = original_builder(method, mixins)
                    if method == 'STCMTrack' and 'evaluation' not in mixins and not {'ltcp', 'ctr'}.intersection(mixins):
                        change(config)
                    return config

                with self.subTest(dataset=dataset, setting=label), \
                        patch.object(mapping, 'build_config', changed_control):
                    rows = mapping.compute(dataset)
                    self.assertNotEqual(rows['baseline']['shared_settings_sha256'], rows['ltcp']['shared_settings_sha256'])
                    self.assertTrue(any('baseline: non-component settings differ' in issue
                                        for issue in mapping.problems(rows)))

    def test_hash_drift_in_any_of_the_eight_rows_is_rejected(self):
        for dataset in ('antiuav410', 'antiuav300'):
            original_rows = mapping.compute(dataset)
            for name in mapping.COMPONENTS:
                for key in ('shared_settings_sha256', 'eval_shared_settings_sha256'):
                    with self.subTest(dataset=dataset, variant=name, fingerprint=key):
                        rows = copy.deepcopy(original_rows)
                        rows[name][key] = 'different'
                        self.assertTrue(any('non-component settings differ' in issue and key in issue
                                            for issue in mapping.problems(rows)))

    def test_main_check_exits_unsuccessfully_for_baseline_drift(self):
        rows = mapping.compute('antiuav410')
        rows['baseline']['shared_settings_sha256'] = 'different'
        with patch.object(mapping, 'compute', return_value=rows), \
                patch.object(sys, 'argv', ['check_variant_mapping.py', '--check']), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
                self.assertRaises(SystemExit) as raised:
            mapping.main()
        self.assertEqual(raised.exception.code, 1)

    def test_all_eight_rows_share_public_defaults_and_prediction_heads(self):
        for dataset in ('antiuav410', 'antiuav300'):
            rows = mapping.compute(dataset)
            reference = rows['ltcp']
            for name, row in rows.items():
                with self.subTest(dataset=dataset, variant=name):
                    self.assertEqual(row['paper_settings'], reference['paper_settings'])
                    self.assertEqual(row['model']['heads'], reference['model']['heads'])
                    self.assertEqual(row['model']['head_definitions_sha256'],
                                     reference['model']['head_definitions_sha256'])

    def test_check_rejects_changes_to_backbone_training_and_post_processing(self):
        original_rows = mapping.compute('antiuav410')
        changes = (
            ('backbone', lambda cfg: cfg['model']['backbone']['parameters'].update(name='ViT-S/14')),
            ('optimizer', lambda cfg: cfg['run']['runner']['train']['optimization']['optimizer'].update(lr=0.001)),
            ('sampling', lambda cfg: cfg['run']['data']['train']['siamese_training_pair_sampling'][
                'positive_sample'].update(num_template_frames=3)),
            ('loss', lambda cfg: cfg['run']['runner']['train']['criteria']['bbox_regression'].update(weight=2.)),
            ('post_process', lambda cfg: cfg['run']['runner']['test']['evaluator']['pipeline'][
                'post_process'].update(window_penalty=0.45)),
        )
        for label, change in changes:
            with self.subTest(setting=label):
                config = self.ablation_config('full')
                change(config)
                rows = copy.deepcopy(original_rows)
                rows['full'].update(mapping.settings_facts(config))
                self.assertTrue(any('non-component settings differ' in problem for problem in mapping.problems(rows)))

    def test_check_rejects_inconsistent_component_parameters(self):
        original_rows = mapping.compute('antiuav410')
        for component in ('ltcp', 'ctr'):
            with self.subTest(component=component):
                config = self.ablation_config('full')
                if component == 'ltcp':
                    config['model']['ltcp']['memory_size'] = 3
                else:
                    config['run']['runner']['test']['evaluator']['pipeline']['ctr']['confidence_threshold'] = 0.5
                rows = copy.deepcopy(original_rows)
                rows['full'].update(mapping.settings_facts(config))
                self.assertTrue(any('different component parameters' in problem for problem in mapping.problems(rows)))

    def test_full_frame_input_is_only_a_ctr_requirement(self):
        rows = mapping.compute('antiuav410')
        rows['ltcp']['full_template_inputs']['test'] = True
        self.assertTrue(any('full-template-image inputs' in problem for problem in mapping.problems(rows)))

    def test_check_rejects_baseline_label_augmentation_and_crop_drift(self):
        changes = (
            ('labels', lambda cfg: cfg['run']['data']['train']['transform']['plugin'][0].update(
                positive_assignment='box', center_positive_radius=1)),
            ('augmentation', lambda cfg: cfg['run']['data']['train']['transform']['augmentation'][0].update(joint=False)),
            ('crop', lambda cfg: cfg['run']['runner']['test']['evaluator']['pipeline'][
                'search_region_cropping'].update(min_object_size=10)),
        )
        for label, change in changes:
            with self.subTest(setting=label):
                cfg = self.ablation_config('baseline')
                change(cfg)
                rows = mapping.compute('antiuav410')
                rows['baseline'].update(mapping.config_facts(cfg))
                self.assertTrue(any('shared public settings differ' in problem for problem in mapping.problems(rows)))

    def test_check_rejects_augmentation_that_misses_a_models_input(self):
        for name in ('baseline', 'full'):
            with self.subTest(variant=name):
                cfg = self.ablation_config(name)
                cfg['run']['data']['train']['transform']['augmentation'][0]['target'].pop()
                rows = mapping.compute('antiuav410')
                rows[name].update(mapping.config_facts(cfg))
                self.assertTrue(any('must target every template and search input' in problem
                                    for problem in mapping.problems(rows)))

    def test_shared_mixins_are_identical_copies(self):
        for name in ('disable_torch_compile', 'evaluation', 'eval_short', 'dataset_antiuav300'):
            with self.subTest(mixin=name):
                self.assertEqual((ROOT / f'config/SPMTrack/_mixin/{name}.yaml').read_bytes(),
                                 (ROOT / f'config/STCMTrack/_mixin/{name}.yaml').read_bytes())


class STCMTrackTrainingEntryTests(unittest.TestCase):
    def test_real_entry_point_builds_the_two_paper_stages_for_both_datasets(self):
        for dataset in ('antiuav410', 'antiuav300'):
            with self.subTest(dataset=dataset):
                stages = mapping.compute_training(dataset)
                self.assertEqual(mapping.training_problems(stages), [])
                self.assertEqual(stages[1]['method'], 'STCMTrack')
                self.assertEqual((stages[1]['epochs'], stages[1]['ltcp_enabled'], stages[1]['train_only']),
                                 (80, False, False))
                self.assertEqual((stages[2]['epochs'], stages[2]['ltcp_enabled'], stages[2]['train_only']),
                                 (20, True, True))
                self.assertIn('ltcp_stage2', stages[2]['mixins'])
                self.assertTrue(stages[2]['checkpoint_required'])
                self.assertTrue(stages[2]['loads_stage1_checkpoint'])
                self.assertTrue(stages[2]['checkpoint_preflight_before_cuda'])

    def test_check_rejects_changed_stage2_yaml_budget_or_freeze_after_real_mixin_application(self):
        original_loader = mapping.load_yaml
        for setting, value in (('run.num_epochs', 10), ('model.ltcp.train_only', False)):
            def changed_stage2(path):
                rules = original_loader(path)
                if Path(path).name == 'ltcp_stage2.yaml':
                    rules = copy.deepcopy(rules)
                    for rule in rules:
                        if rule['path'] == setting:
                            rule['value'] = value
                return rules

            with self.subTest(setting=setting), patch.object(mapping, 'load_yaml', changed_stage2):
                stages = mapping.compute_training('antiuav410')
                self.assertTrue(any('only LTCP for 20 epochs' in problem
                                    for problem in mapping.training_problems(stages)))

    def test_check_rejects_wrong_training_stage_mixin_wiring(self):
        original = mapping.TRAIN_SCRIPT.read_text(encoding='utf-8')
        wrong_scripts = (
            original.replace('mixin_names+=(ltcp ltcp_stage2)', 'mixin_names+=(ltcp)'),
            original.replace('exp_name="STCMTrack-Train-Stage1-${DATASET}"',
                             'mixin_names+=(ltcp)\n    exp_name="STCMTrack-Train-Stage1-${DATASET}"'),
        )
        for number, text in enumerate(wrong_scripts):
            with self.subTest(case=number):
                self.assertTrue(mapping.training_problems(mapping.compute_training(text=text)))

    def test_check_rejects_missing_checkpoint_handoff_or_preflight(self):
        original = mapping.TRAIN_SCRIPT.read_text(encoding='utf-8')
        preflight = 'python3 "$REPO_ROOT/tools/check_stcmtrack_weights.py" --base "$BASE_WEIGHT"'
        for removed in ('weight_args+=(--weight_path "$BASE_WEIGHT")',
                        '${weight_args[@]+"${weight_args[@]}"}', preflight):
            with self.subTest(removed=removed):
                stages = mapping.compute_training(text=original.replace(removed, ''))
                self.assertTrue(any('checkpoint' in problem for problem in mapping.training_problems(stages)))

    def test_hard_bce_targets_are_shared_defaults_not_an_explicit_paper_requirement(self):
        settings = mapping.compute_training()[1]['shared_public_settings']
        settings['criteria']['classification']['iou_aware_classification_score'] = True
        self.assertEqual(mapping.explicit_paper_setting_problems(settings), [])


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

    def test_training_keeps_spm_structure_and_uses_shared_loss_defaults(self):
        train = self.cfg['run']['data']['train']
        positive = train['siamese_training_pair_sampling']['positive_sample']
        self.assertEqual((positive['sample_mode'], positive['num_template_frames'], positive['num_search_frames'],
                          positive['MAX_SAMPLE_INTERVAL']), ('interval', 3, 2, 400))
        flip, color, deit = train['transform']['augmentation']
        self.assertEqual(flip['target'], ['template_0', 'template_1', 'template_2', 'search_region_0', 'search_region_1'])
        self.assertTrue(flip['joint'])
        self.assertTrue(color['joint'])
        self.assertTrue(deit['joint'])
        self.assertTrue(train['transform']['temporal_consistent_crops'])
        label_plugin = train['transform']['plugin'][0]
        self.assertEqual((label_plugin['positive_assignment'], label_plugin['center_positive_radius']), ('center', 0))
        criteria = self.cfg['run']['runner']['train']['criteria']
        self.assertEqual(criteria.get('frame_loss_reduction', 'mean'), 'mean')
        self.assertFalse(criteria['classification']['iou_aware_classification_score'])

    def test_evaluation_keeps_the_independent_model_with_shared_crop_and_window(self):
        pipeline = self.cfg['run']['runner']['test']['evaluator']['pipeline']
        self.assertEqual(pipeline['type'], 'spmtrack_one_stream_tracker')
        self.assertNotIn('ctr', pipeline)
        self.assertEqual(pipeline['search_region_cropping'], {'type': 'simple', 'min_object_size': 1, 'area_factor': 5.0})
        self.assertEqual(pipeline['post_process'], {'type': 'box_with_score_map_spmtrack', 'window_penalty': 0.0})
        self.assertEqual(pipeline['online_template']['area_factor'], 2.0)
        self.assertEqual([p['type'] for p in pipeline['plugin']], ['template_foreground_indicating_mask_generation'])

    def test_evaluation_uses_the_shared_metric_handlers(self):
        for task in ('test', 'eval'):
            handlers = self.cfg['run']['data'][task]['result_collector']['dispatch'][0]['handlers']
            self.assertEqual(handlers[0]['type'], 'one_pass_evaluation_compatible')

    def test_training_budget_is_shared_with_stcmtrack(self):
        self.assertEqual(self.cfg['run']['num_epochs'], 80)
        self.assertEqual(self.cfg['run']['data']['train']['global_batch_size'], 4)
        self.assertFalse(self.cfg['run']['runner']['train']['torch_compile']['enabled'])
        self.assertFalse(self.cfg['run']['efficiency_assessment']['enabled'])

    def test_optimizer_and_losses_share_the_public_configuration(self):
        stcm = load_yaml(str(ROOT / 'config/STCMTrack/dinov2/config.yaml'))
        self.assertEqual(mapping.paper_settings(self.cfg), mapping.paper_settings(stcm))
        optimization = self.cfg['run']['runner']['train']['optimization']
        self.assertNotIn('per_parameter', optimization['optimizer'])
        self.assertEqual(optimization['lr_scheduler']['parameters']['warmup_epochs'], 0)
        self.assertEqual(optimization['lr_scheduler']['parameters']['lr_min'], 0.)

    def test_stcmtrack_config_keeps_its_own_definitions(self):
        stcm = load_yaml(str(ROOT / 'config/STCMTrack/dinov2/config.yaml'))
        pipeline = stcm['run']['runner']['test']['evaluator']['pipeline']
        self.assertEqual(stcm['type'], 'STCMTrack')
        self.assertEqual(pipeline['type'], 'one_stream_tracker')
        self.assertEqual(pipeline['post_process'], {'type': 'box_with_score_map', 'window_penalty': 0.0})
        positive = stcm['run']['data']['train']['siamese_training_pair_sampling']['positive_sample']
        self.assertEqual((positive['sample_mode'], positive['num_template_frames'], positive['num_search_frames']),
                         ('first_frame_causal', 1, 3))
        self.assertEqual(stcm['run']['runner']['train']['criteria']['frame_loss_reduction'], 'mean')


class IsolationTests(unittest.TestCase):
    def test_independent_spmtrack_sources_do_not_import_stcmtrack_ltcp_or_ctr(self):
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

    def test_independent_spmtrack_text_does_not_instantiate_ltcp_or_ctr(self):
        for entry in SPMTRACK_SOURCES:
            for path in python_files(entry):
                code = re.sub(r'"""(.|\n)*?"""|#.*', '', path.read_text(encoding='utf-8'))
                with self.subTest(file=str(path.relative_to(ROOT))):
                    self.assertIsNone(re.search(r'LocalEnhancedTemporalContextPropagation|ConfidenceTriggeredRelocalization|'
                                                r'LTCPConfig|CTRConfig', code))


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


class WeightCheckTests(unittest.TestCase):
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
        self.assertIn('--allow-unmarked', result['provenance'])

    def test_stcmtrack_keys_are_always_rejected(self):
        for key in ('_expert_alpha', '_use_rsexpert', 'ltcp.gate.weight'):
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
