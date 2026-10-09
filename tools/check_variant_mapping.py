#!/usr/bin/env python3
"""Print the model and the components that every `VARIANT` of test_stcmtrack.sh selects.

The table is derived statically, without importing torch, from
  * the `case "$VARIANT"` block of test_stcmtrack.sh (method, mixins),
  * the YAML configs loaded with the same mixin search path and order as boot.sh,
  * the AST of the registries and builders: type -> builder -> returned classes -> base classes,
  * the transitive import closure of the model builder and of the evaluation-pipeline builder, which shows
    whether LTCP / CTR / STCMTrack code is reachable.

  python tools/check_variant_mapping.py            # markdown table
  python tools/check_variant_mapping.py --json
  python tools/check_variant_mapping.py --check    # also check the paper's two-stage STCMTrack training entry point
"""
import argparse
import ast
import copy
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trackit.core.boot.funcs.utils.custom_yaml_loader import load_yaml  # noqa: E402
from trackit.core.boot.funcs.mixin import apply_static_mixin_rules  # noqa: E402

SCRIPT = ROOT / 'test_stcmtrack.sh'
TRAIN_SCRIPT = ROOT / 'train_stcmtrack.sh'
CONFIG_ROOT = ROOT / 'config'
CONFIG_NAME = 'dinov2'
# Table 2 row 1 is the independent SPMTrack baseline; rows 2-8 are STCMTrack configurations.
# The mapping does not infer that the published scores were produced by the current code.
COMPONENTS = {
    'baseline': (False, False, False), 'ltcp': (True, False, False), 'mcc': (False, True, False),
    'rgtc': (False, False, True), 'ltcp_mcc': (True, True, False), 'ltcp_rgtc': (True, False, True),
    'mcc_rgtc': (False, True, True), 'full': (True, True, True),
}
STCM_VARIANTS = tuple(name for name in COMPONENTS if name != 'baseline')


# ----------------------------------------------------------------------------- shell
def parse_case_block(text, variable):
    match = re.search(r'case "\$' + variable + r'" in\n(.*?)\nesac', text, re.S)
    if not match:
        raise ValueError(f'case "${variable}" block not found in {SCRIPT.name}')
    entries = {}
    # A branch may span several lines (DATASET) or sit on one line (VARIANT); `*)` defaults are skipped.
    for m in re.finditer(r'^[ \t]*([A-Za-z0-9_]+(?:\|[A-Za-z0-9_]+)*)\)(.*?);;', match.group(1), re.S | re.M):
        entries[tuple(m.group(1).split('|'))] = m.group(2).strip()
    return entries


def parse_variants(text):
    default_method = re.search(r'^method_name=(\w+)', text, re.M).group(1)
    variants = {}
    for names, body in parse_case_block(text, 'VARIANT').items():
        method = re.search(r'method_name=(\w+)', body)
        mixins = re.search(r'variant_mixins=\(([^)]*)\)', body).group(1).split()
        uses_ltcp = 'use_ltcp=true' in body
        for name in names:
            variants[name] = {'aliases': list(names), 'method': method.group(1) if method else default_method,
                              'variant_mixins': mixins, 'script_use_ltcp': uses_ltcp}
    dataset_mixins = {}
    for names, body in parse_case_block(text, 'DATASET').items():
        mixins = re.search(r'dataset_mixins=\(([^)]*)\)', body)
        for name in names:
            if mixins is not None:
                dataset_mixins[name] = mixins.group(1).split()
    if not re.search(r'mixin_names=\(disable_torch_compile .*dataset_mixins.*variant_mixins.* evaluation\)', text):
        raise ValueError('mixin order in test_stcmtrack.sh changed; update this tool')
    return variants, dataset_mixins


# ----------------------------------------------------------------------------- config
def mixin_path(method, name):
    for candidate in (CONFIG_ROOT / method / CONFIG_NAME / 'mixin' / f'{name}.yaml',
                      CONFIG_ROOT / method / '_mixin' / f'{name}.yaml',
                      CONFIG_ROOT / '_mixin' / f'{name}.yaml'):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f'mixin {name!r} not found for method {method}')


def build_config(method, mixins):
    config = load_yaml(str(CONFIG_ROOT / method / CONFIG_NAME / 'config.yaml'))
    for name in mixins:
        apply_static_mixin_rules(load_yaml(str(mixin_path(method, name))), config)
    return config


def config_facts(config):
    pipeline = config['run']['runner']['test']['evaluator']['pipeline']
    ctr = pipeline.get('ctr')
    ctr_on = bool(ctr and ctr.get('enabled', False))
    train = config['run']['data']['train']
    positive = train['siamese_training_pair_sampling']['positive_sample']
    criteria = copy.deepcopy(config['run']['runner']['train']['criteria'])
    criteria.setdefault('frame_loss_reduction', 'mean')
    return {
        'config_type': config['type'],
        'ltcp': bool(config['model'].get('ltcp', {}).get('enabled', False)),
        'mcc': ctr_on and bool(ctr.get('mcc_enabled', True)),
        'rgtc': ctr_on and bool(ctr.get('rgtc_enabled', True)),
        'train_templates': positive['num_template_frames'],
        'train_search_frames': positive['num_search_frames'],
        'train_sample_mode': positive['sample_mode'],
        'pipeline_type': pipeline['type'],
        'post_process': pipeline['post_process']['type'],
        'window_penalty': pipeline['post_process']['window_penalty'],
        'frame_loss_reduction': criteria.get('frame_loss_reduction', 'mean'),
        'iou_aware_classification': criteria['classification']['iou_aware_classification_score'],
        'stage1_epochs': config['run']['num_epochs'],
        'paper_settings': paper_settings(config),
        'augmentation_target_problems': augmentation_target_problems(config),
        'full_template_inputs': {task: bool(config['run']['data'][task]['transform'].get(
            'with_full_template_image', False)) for task in ('test', 'eval')},
    }


def augmentation_target_problems(config):
    """A shared augmentation must cover every input of its model, even when counts differ."""
    found = []
    for task in ('train', 'val'):
        data = config['run']['data'][task]
        positive = data['siamese_training_pair_sampling']['positive_sample']
        expected = {f'template_{index}' for index in range(positive['num_template_frames'])} | {
            f'search_region_{index}' for index in range(positive['num_search_frames'])}
        for augmentation in data['transform'].get('augmentation', []):
            targets = augmentation.get('target', [])
            if set(targets) != expected or len(targets) != len(expected):
                found.append(f'{task}: {augmentation["type"]} must target every template and search input once')
    return found


def training_data_settings(config):
    """Compare external training conditions without equating the models' temporal inputs."""
    settings = {}
    for task in ('train', 'val'):
        data = copy.deepcopy(config['run']['data'][task])
        data.pop('siamese_training_pair_sampling')
        transform = data['transform']
        transform.setdefault('temporal_consistent_crops', False)
        for augmentation in transform.get('augmentation', []):
            # Targets are checked against each model's actual input counts separately.
            augmentation.pop('target', None)
            augmentation.setdefault('joint', True)
        settings[task] = data
    return settings


def evaluation_settings(config):
    """Shared crop, data and engine settings; pipeline/post-process classes stay independent."""
    test = config['run']['runner']['test']
    pipeline = test['evaluator']['pipeline']
    cropping = copy.deepcopy(pipeline['search_region_cropping'])
    minimum = cropping.get('min_object_size', 1.)
    cropping['min_object_size'] = [float(value) for value in minimum] if isinstance(minimum, (list, tuple)) \
        else [float(minimum), float(minimum)]
    data_settings = {}
    for task in ('test', 'eval'):
        if task not in config['run']['data']:
            continue
        data = copy.deepcopy(config['run']['data'][task])
        # Full-frame inputs are needed by MCC/RGTC, rather than an evaluation protocol difference.
        data['transform'].pop('with_full_template_image', None)
        data_settings[task] = data
    return {'data': data_settings, 'inference_engine': copy.deepcopy(test['inference_engine']),
            'search_region_cropping': cropping,
            'window_penalty': pipeline['post_process']['window_penalty']}


def paper_settings(config):
    """Shared public settings, including defaults not specified by the paper.

    The historical name is retained for consumers of the mapping JSON. Only the
    subset checked by explicit_paper_setting_problems is asserted as Sec. 3.1.
    """
    optimization = config['run']['runner']['train']['optimization']
    criteria = copy.deepcopy(config['run']['runner']['train']['criteria'])
    criteria.setdefault('frame_loss_reduction', 'mean')
    return {
        'backbone': copy.deepcopy(config['model']['backbone']),
        'template_size': config['common']['template_size'],
        'search_region_size': config['common']['search_region_size'],
        'response_map_size': config['common']['response_map_size'],
        'common_inputs': copy.deepcopy(config['common']),
        'tmoe': copy.deepcopy(config['model']['tmoe']),
        'stage1_epochs': config['run']['num_epochs'],
        'optimizer': copy.deepcopy(optimization['optimizer']),
        'scheduler': copy.deepcopy(optimization['lr_scheduler']),
        'optimization': copy.deepcopy(optimization),
        'torch_compile': copy.deepcopy(config['run']['runner']['train']['torch_compile']),
        'training_data': training_data_settings(config),
        'evaluation': evaluation_settings(config),
        'criteria': copy.deepcopy(criteria),
        'metric_handlers': {
            task: [handler['type'] for handler in config['run']['data'][task][
                'result_collector']['dispatch'][0]['handlers']]
            for task in ('test', 'eval')
        },
    }


def shared_config(config):
    """Keep every setting except the named components and their necessary full-frame input.

    Component parameters are checked separately, so removing their dictionaries here does not
    permit changes to their thresholds or memory settings between ablation rows.
    """
    common = copy.deepcopy(config)
    common.pop('name', None)
    common.pop('logging', None)
    common['model'].pop('ltcp', None)
    common['run']['runner']['test']['evaluator']['pipeline'].pop('ctr', None)
    for task in ('test', 'eval'):
        common['run']['data'].get(task, {}).get('transform', {}).pop('with_full_template_image', None)
    return common


def fingerprint(value):
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def settings_facts(config):
    ltcp = copy.deepcopy(config['model'].get('ltcp', {}))
    ctr = copy.deepcopy(config['run']['runner']['test']['evaluator']['pipeline'].get('ctr', {}))
    ltcp.pop('enabled', None)
    for key in ('enabled', 'mcc_enabled', 'rgtc_enabled'):
        ctr.pop(key, None)
    return {'shared_settings_sha256': fingerprint(shared_config(config)),
            'ltcp_parameters_sha256': fingerprint(ltcp) if ltcp else None,
            'ctr_parameters_sha256': fingerprint(ctr) if ctr else None,
            'ltcp_parameters': ltcp or None, 'ctr_parameters': ctr or None}


def parse_training_stages(text):
    """Read the supported training entry point without executing conda or CUDA.

    Unknown shell layouts fail closed so a changed mixin branch cannot silently
    be checked against hard-coded stage assumptions.
    """
    stage = re.search(r'^if \[\[ "\$TRAIN_STAGE" == 1 \]\]; then\n(.*?)\nelse\n(.*?)\nfi', text, re.S | re.M)
    initial = re.search(r'^mixin_names=\(([^\n]*)\)$', text, re.M)
    method = re.search(r'^"\$REPO_ROOT/boot\.sh" (\w+) (\w+) ', text, re.M)
    if stage is None or initial is None or method is None:
        raise ValueError('training stage layout changed; update check_variant_mapping.py')
    common_mixins = initial.group(1).split()
    if common_mixins != ['disable_torch_compile', '${dataset_mixins[@]+"${dataset_mixins[@]}"}']:
        raise ValueError('training mixin initialization changed; update check_variant_mapping.py')
    if method.groups() != ('STCMTrack', CONFIG_NAME):
        raise ValueError('the two-stage training entry point must select STCMTrack dinov2')
    append_matches = list(re.finditer(r'^\s*mixin_names\+=\(([^)]*)\)', text, re.M))
    stages = {}
    for number, body in enumerate(stage.groups(), 1):
        mixins = []
        for match in re.finditer(r'^\s*mixin_names\+=\(([^)]*)\)', body, re.M):
            values = match.group(1).split()
            if any(re.fullmatch(r'\w+', value) is None for value in values):
                raise ValueError('dynamic training stage mixins require an updated checker')
            mixins.extend(values)
        stages[number] = {'method': method.group(1), 'mixins': ['disable_torch_compile'] + mixins}
    if len(append_matches) != sum(len(re.findall(r'^\s*mixin_names\+=', body, re.M))
                                  for body in stage.groups()):
        raise ValueError('training mixins outside the stage branches require an updated checker')
    stage2 = stage.group(2)
    stages[2]['checkpoint_required'] = bool(re.search(
        r'if \[\[ -z "\$BASE_WEIGHT" \|\| ! -f "\$BASE_WEIGHT" \]\]; then.*?exit 1\s*fi', stage2, re.S))
    stages[2]['loads_stage1_checkpoint'] = 'weight_args+=(--weight_path "$BASE_WEIGHT")' in stage2 \
        and '${weight_args[@]+"${weight_args[@]}"}' in text[method.start():]
    preflight = 'python3 "$REPO_ROOT/tools/check_stcmtrack_weights.py" --base "$BASE_WEIGHT"'
    stages[2]['checkpoint_preflight_before_cuda'] = preflight in stage2 \
        and text.find(preflight) < text.find('python3 - "$first_device_id"')
    dataset_mixins = {}
    for names, body in parse_case_block(text, 'DATASET').items():
        match = re.search(r'dataset_mixins=\(([^)]*)\)', body)
        for name in names:
            if match is not None:
                dataset_mixins[name] = match.group(1).split()
    return stages, dataset_mixins


def compute_training(dataset='antiuav410', text=None):
    stages, datasets = parse_training_stages(TRAIN_SCRIPT.read_text(encoding='utf-8') if text is None else text)
    for info in stages.values():
        info['mixins'] = info['mixins'][:1] + datasets[dataset] + info['mixins'][1:]
        config = build_config(info['method'], info['mixins'])
        ltcp = config['model'].get('ltcp', {})
        info.update(epochs=config['run']['num_epochs'], ltcp_enabled=bool(ltcp.get('enabled', False)),
                    train_only=bool(ltcp.get('train_only', False)), shared_public_settings=paper_settings(config))
    return stages


def explicit_paper_setting_problems(settings, epochs=80):
    """Check what Sec. 3.1 actually specifies, excluding label and scheduler defaults."""
    optimizer, criteria = settings['optimizer'], settings['criteria']
    expected = (
        settings['backbone']['type'] == 'DINOv2',
        settings['backbone']['parameters']['name'] == 'ViT-B/14',
        settings['template_size'] == [196, 196],
        settings['search_region_size'] == [378, 378],
        settings['stage1_epochs'] == epochs,
        optimizer['type'] == 'AdamW', optimizer['lr'] == 1.e-4, optimizer['weight_decay'] == .1,
        settings['scheduler']['sched'] == 'cosine',
        criteria['classification']['type'] == 'binary_cross_entropy',
        criteria['bbox_regression']['type'] == 'GIoU',
        criteria['classification']['weight'] == criteria['bbox_regression']['weight'] == 1.,
    )
    return [] if all(expected) else ['configuration differs from the explicit Sec. 3.1 settings']


def training_problems(stages):
    """This 80+20 contract belongs to STCMTrack, not the independent SPMTrack baseline."""
    found = []
    stage1, stage2 = stages[1], stages[2]
    if stage1['epochs'] != 80 or stage1['ltcp_enabled'] or stage1['train_only']:
        found.append('STCMTrack stage 1 must train the tracker for 80 epochs with LTCP disabled')
    if stage2['epochs'] != 20 or not stage2['ltcp_enabled'] or not stage2['train_only']:
        found.append('STCMTrack stage 2 must train only LTCP for 20 epochs')
    if not stage2['checkpoint_required'] or not stage2['loads_stage1_checkpoint']:
        found.append('STCMTrack stage 2 must require and load the supplied stage-1 BASE_WEIGHT checkpoint')
    if not stage2['checkpoint_preflight_before_cuda']:
        found.append('STCMTrack stage 2 must check the stage-1 checkpoint before CUDA initialization')
    for number, info in stages.items():
        found.extend(f'STCMTrack stage {number}: {problem}' for problem in
                     explicit_paper_setting_problems(info['shared_public_settings'], epochs=80 if number == 1 else 20))
    return found


# ----------------------------------------------------------------------------- AST
def module_file(module):
    base = ROOT.joinpath(*module.split('.'))
    if base.with_suffix('.py').is_file():
        return base.with_suffix('.py')
    if (base / '__init__.py').is_file():
        return base / '__init__.py'
    return None


def file_module(path):
    parts = list(path.relative_to(ROOT).with_suffix('').parts)
    if parts[-1] == '__init__':
        parts.pop()
    return '.'.join(parts)


def absolute_import_base(path, node):
    if not node.level:
        return node.module or ''
    package = file_module(path) if path.name == '__init__.py' else file_module(path).rpartition('.')[0]
    parts = package.split('.')
    parts = parts[:len(parts) - (node.level - 1)]
    return '.'.join(parts + ([node.module] if node.module else []))


def imported_modules(path):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = absolute_import_base(path, node)
            names = [base] + [f'{base}.{alias.name}' for alias in node.names]
        for name in names:
            if name.startswith('trackit') and module_file(name) is not None:
                yield name


def import_closure(start_file):
    seen, queue = set(), [file_module(start_file)]
    while queue:
        module = queue.pop()
        if module in seen:
            continue
        seen.add(module)
        parts = module.split('.')
        for i in range(1, len(parts)):  # importing a.b.c executes a/__init__ and a/b/__init__
            queue.append('.'.join(parts[:i]))
        file = module_file(module)
        if file is not None:
            queue.extend(imported_modules(file))
    return seen


def dispatch_table(path, subject):
    """`if X['type'] == 'name': from .pkg.builder import fn` -> {name: 'absolute.module'}."""
    table = {}
    for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare) and len(node.test.comparators) == 1 \
                and isinstance(node.test.comparators[0], ast.Constant) and isinstance(node.test.comparators[0].value, str):
            left = ast.unparse(node.test.left)
            if left in subject:
                for stmt in node.body:
                    if isinstance(stmt, ast.ImportFrom):
                        table[node.test.comparators[0].value] = absolute_import_base(path, stmt)
                        break
    return table


def returned_classes(builder_file, package):
    """Classes of `package` that the builder instantiates (imported names that are called)."""
    tree = ast.parse(builder_file.read_text(encoding='utf-8'))
    imported = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            base = absolute_import_base(builder_file, node)
            for alias in node.names:
                imported[alias.asname or alias.name] = base
    called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    return {name: module for name, module in imported.items()
            if name in called and name[:1].isupper() and module.startswith(package + '.')}


def class_bases(module, class_name):
    file = module_file(module)
    for node in ast.walk(ast.parse(file.read_text(encoding='utf-8'))):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return [ast.unparse(base) for base in node.bases]
    return None


def main_pipeline_class(builder_file):
    for node in ast.walk(ast.parse(builder_file.read_text(encoding='utf-8'))):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'main_pipeline' for t in node.targets) \
                and isinstance(node.value, ast.Call):
            return ast.unparse(node.value.func)
    return None


def code_reach(closure):
    components = [set(module.split('.')) for module in closure]
    return {'ltcp_code': any('ltcp' in c for c in components),
            'ctr_code': any('ctr' in c for c in components),
            'stcmtrack_package': any(m.startswith('trackit.models.methods.STCMTrack') for m in closure)}


def source_structure_sha256(path):
    """Ignore comments and documentation when comparing the shared prediction-head implementation."""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.body and isinstance(node.body[0], ast.Expr) \
                and isinstance(node.body[0].value, ast.Constant) \
                and isinstance(node.body[0].value.value, str):
            node.body.pop(0)
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode('utf-8')).hexdigest()


def resolve_model(config_type):
    registry = dispatch_table(ROOT / 'trackit/models/methods/builder.py', {"config['type']"})
    builder_module = registry[config_type]
    builder_file = module_file(builder_module)
    classes = returned_classes(builder_file, builder_module.rpartition('.')[0])
    chains = {name: class_bases(module, name) for name, module in classes.items()}
    heads, definitions = [], []
    for name, module in classes.items():
        if 'Inference' in name:
            continue
        source = module_file(module)
        tree = ast.parse(source.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Attribute) and ast.unparse(target) == 'self.head'
                    for target in node.targets) and isinstance(node.value, ast.Call):
                heads.append(ast.unparse(node.value))
                if isinstance(node.value.func, ast.Name):
                    for imported in ast.walk(tree):
                        if isinstance(imported, ast.ImportFrom) and any(
                                (alias.asname or alias.name) == node.value.func.id for alias in imported.names):
                            definition = module_file(absolute_import_base(source, imported))
                            definitions.append(source_structure_sha256(definition))
    return {'builder': builder_module, 'classes': sorted(classes), 'bases': chains,
            'heads': sorted(set(heads)), 'head_definitions_sha256': sorted(set(definitions)),
            'reach': code_reach(import_closure(builder_file))}


def resolve_pipeline(pipeline_type):
    registry_file = ROOT / 'trackit/runner/evaluation/distributed/tracker_evaluator/default/pipelines/builder.py'
    registry = dispatch_table(registry_file, {"pipeline_config['type']"})
    builder_module = registry[pipeline_type]
    builder_file = module_file(builder_module)
    return {'builder': builder_module, 'main_class': main_pipeline_class(builder_file),
            'reach': code_reach(import_closure(builder_file))}


# ----------------------------------------------------------------------------- mapping
def compute(dataset='antiuav410'):
    text = SCRIPT.read_text(encoding='utf-8')
    variants, dataset_mixins = parse_variants(text)
    rows = {}
    for name, info in variants.items():
        base_mixins = ['disable_torch_compile'] + dataset_mixins[dataset] + info['variant_mixins']
        config = build_config(info['method'], base_mixins)
        eval_config = build_config(info['method'], base_mixins + ['evaluation'])
        facts = config_facts(config)
        model = resolve_model(facts['config_type'])
        pipeline = resolve_pipeline(facts['pipeline_type'])
        rows[name] = {**info, **facts, **settings_facts(config),
                      'eval_shared_settings_sha256': settings_facts(eval_config)['shared_settings_sha256'],
                      'eval_public_settings_sha256': fingerprint(evaluation_settings(eval_config)),
                      'model': model, 'pipeline': pipeline}
    return rows


def problems(rows):
    found = []
    if set(rows) != set(COMPONENTS):
        found.append(f'Expected exactly the eight Table 2 variants, got {sorted(rows)}')
    for name, row in rows.items():
        found.extend(f'{name}: {problem}' for problem in row['augmentation_target_problems'])
        if row['method'] != row['config_type']:
            found.append(f'{name}: script method {row["method"]} but config type {row["config_type"]}')
        if row['script_use_ltcp'] != row['ltcp']:
            found.append(f'{name}: script use_ltcp={row["script_use_ltcp"]} but config LTCP={row["ltcp"]}')
        if row['model']['heads'] != ['MlpAnchorFreeHead(self.embed_dim, self.x_size)'] \
                or len(row['model']['head_definitions_sha256']) != 1:
            found.append(f'{name}: expected the shared center/box MLP prediction head')
        if name in COMPONENTS and (row['ltcp'], row['mcc'], row['rgtc']) != COMPONENTS[name]:
            found.append(f'{name}: components {(row["ltcp"], row["mcc"], row["rgtc"])} instead of '
                         f'{COMPONENTS[name]}')
        if row['method'] == 'SPMTrack':
            if (row['ltcp'], row['mcc'], row['rgtc']) != (False, False, False):
                found.append(f'{name}: the SPMTrack baseline must have LTCP/MCC/RGTC off')
            reach = {**row['model']['reach'], **{'pipeline_' + k: v for k, v in row['pipeline']['reach'].items()}}
            for key in ('ltcp_code', 'ctr_code', 'stcmtrack_package', 'pipeline_ltcp_code', 'pipeline_ctr_code'):
                if reach[key]:
                    found.append(f'{name}: SPMTrack code path can reach {key}')
            if row['model']['classes'] != ['SPMTrackInference_DINOv2', 'SPMTrack_DINOv2']:
                found.append(f'{name}: SPMTrack builder returns {row["model"]["classes"]}')
            if row['model']['bases'].get('SPMTrackInference_DINOv2') != ['SPMTrack_DINOv2']:
                found.append(f'{name}: SPMTrackInference_DINOv2 must derive from SPMTrack_DINOv2 only')
            if row['pipeline']['main_class'] != 'SPMTrackOneStream_Evaluation_MainPipeline':
                found.append(f'{name}: SPMTrack evaluation pipeline is {row["pipeline"]["main_class"]}')
            if (row['train_templates'], row['train_search_frames']) != (3, 2):
                found.append(f'{name}: SPMTrack training must use 3 templates and 2 search frames')
        else:
            if row['train_templates'] != 1:
                found.append(f'{name}: STCMTrack variants use one template, got {row["train_templates"]}')
            if row['pipeline_type'] != 'one_stream_tracker':
                found.append(f'{name}: unexpected pipeline {row["pipeline_type"]}')
            if row['post_process'] != 'box_with_score_map' or row['window_penalty'] != 0.:
                found.append(f'{name}: STCMTrack post-processing must be the no-Hann definition')
    spm_names = {n for n, r in rows.items() if r['method'] == 'SPMTrack'}
    if spm_names != {'baseline'}:
        found.append(f'Table 2 row 1 must be the independent SPMTrack baseline, got {sorted(spm_names)}')
    controlled = [rows[name] for name in STCM_VARIANTS if name in rows]
    if len(controlled) != len(STCM_VARIANTS):
        found.append('All seven STCMTrack configurations (Table 2 rows 2-8) must exist')
    if controlled:
        reference = controlled[0]
        for name in STCM_VARIANTS:
            if name not in rows:
                continue
            row = rows[name]
            if row['method'] != 'STCMTrack' or row['model'] != reference['model'] \
                    or row['pipeline'] != reference['pipeline']:
                found.append(f'{name}: Table 2 rows 2-8 must use the same STCMTrack model and pipeline')
            for key in ('shared_settings_sha256', 'eval_shared_settings_sha256'):
                if row[key] != reference[key]:
                    found.append(f'{name}: non-component settings differ from ltcp ({key})')
            needs_full_frame = row['mcc'] or row['rgtc']
            if any(value != needs_full_frame for value in row['full_template_inputs'].values()):
                found.append(f'{name}: full-template-image inputs must be enabled exactly when MCC or RGTC is used')
        for key in ('ltcp_parameters_sha256', 'ctr_parameters_sha256'):
            if len({row[key] for row in controlled if row[key] is not None}) > 1:
                found.append(f'Ablations use different component parameters ({key})')
        for name, row in rows.items():
            if row['paper_settings'] != reference['paper_settings']:
                found.append(f'{name}: shared public settings differ from ltcp')
            if row['eval_public_settings_sha256'] != reference['eval_public_settings_sha256']:
                found.append(f'{name}: shared evaluation settings differ from ltcp')
            if row['model']['heads'] != reference['model']['heads'] or row['model'][
                    'head_definitions_sha256'] != reference['model']['head_definitions_sha256']:
                found.append(f'{name}: prediction head construction differs from ltcp')
    for name, row in rows.items():
        settings = row['paper_settings']
        found.extend(f'{name}: {problem}' for problem in explicit_paper_setting_problems(settings))
        ltcp, ctr = row['ltcp_parameters'], row['ctr_parameters']
        if ltcp and (ltcp['memory_size'] != 2 or ltcp.get('store_enhanced_memory', False)):
            found.append(f'{name}: LTCP must keep the two most recent raw search-token frames')
        if ctr and (ctr['confidence_threshold'] != .40 or ctr['ransac_reproj_threshold'] != 2.0
                    or ctr.get('foreground_mask_mode', 'mog2_residual_union') != 'mog2_residual_union'):
            found.append(f'{name}: CTR configuration differs from the Sec. 2.3 settings')
    return found


def ablation_conflicts(rows):
    """Report the original SPMTrack differences, not silently relabel them as component switches."""
    if 'baseline' not in rows or 'ltcp' not in rows:
        return []
    baseline, stcm = rows['baseline'], rows['ltcp']
    fields = ('train_templates', 'train_search_frames', 'train_sample_mode', 'pipeline_type',
              'post_process', 'window_penalty')
    return [f'Row 1 SPMTrack {field}={baseline[field]!r}; rows 2-8 STCMTrack {field}={stcm[field]!r}'
            for field in fields if baseline[field] != stcm[field]]


def markdown(rows):
    out = ['| VARIANT | method | model class (train / inference) | LTCP | MCC | RGTC | train templates / search frames | '
           'post-process | eval pipeline | LTCP/CTR code reachable |',
           '|---|---|---|:---:|:---:|:---:|---|---|---|---|']
    mark = lambda value: 'on' if value else 'off'
    for name, row in rows.items():
        classes = row['model']['classes']
        train_cls = next(c for c in classes if 'Inference' not in c)
        infer_cls = next(c for c in classes if 'Inference' in c)
        reach = row['model']['reach']
        pipeline_reach = row['pipeline']['reach']
        reachable = 'none' if not (reach['ltcp_code'] or pipeline_reach['ctr_code']) else \
            ', '.join(k for k, v in (('LTCP module', reach['ltcp_code']), ('CTR module', pipeline_reach['ctr_code'])) if v)
        out.append(f"| `{name}` | {row['method']} | `{train_cls}` / `{infer_cls}` | {mark(row['ltcp'])} | {mark(row['mcc'])} | "
                   f"{mark(row['rgtc'])} | {row['train_templates']} / {row['train_search_frames']} "
                   f"({row['train_sample_mode']}) | `{row['post_process']}` (Hann {row['window_penalty']}) | "
                   f"`{row['pipeline']['main_class']}` | {reachable} |")
    out.append('\nAll eight rows share the external training and evaluation settings. '
               'Row 1 is independent SPMTrack; only rows 2-8 share the complete non-component configuration.')
    out.extend(f'- {conflict}' for conflict in ablation_conflicts(rows))
    return '\n'.join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', default='antiuav410', choices=('antiuav410', 'antiuav300'))
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    rows = compute(args.dataset)
    stages = compute_training(args.dataset)
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        print(markdown(rows))
        print('\nSTCMTrack training entry point (independent SPMTrack is excluded):')
        print('| Stage | Epochs | LTCP enabled | Only LTCP trainable |')
        print('|---|---:|:---:|:---:|')
        for number, stage in stages.items():
            print(f'| {number} | {stage["epochs"]} | {stage["ltcp_enabled"]} | {stage["train_only"]} |')
    if args.check:
        found = problems(rows) + training_problems(stages)
        for line in found:
            print('MISMATCH:', line, file=sys.stderr)
        if found:
            raise SystemExit(1)
        print(f'\nvariant mapping and STCMTrack training checks passed for {len(rows)} variant names '
              f'({args.dataset})', file=sys.stderr)


if __name__ == '__main__':
    main()
