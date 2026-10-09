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
  python tools/check_variant_mapping.py --check    # exit 1 if a VARIANT does not select the components listed below
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
CONFIG_ROOT = ROOT / 'config'
CONFIG_NAME = 'dinov2'
# The eight module combinations of Table 2. Row 1 uses the same STCMTrack base as rows 2-8;
# the independent SPMTrack comparison is deliberately separate. No published score is inferred.
COMPONENTS = {
    'baseline': (False, False, False), 'ltcp': (True, False, False), 'mcc': (False, True, False),
    'rgtc': (False, False, True), 'ltcp_mcc': (True, True, False), 'ltcp_rgtc': (True, False, True),
    'mcc_rgtc': (False, True, True), 'full': (True, True, True),
}


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
    criteria = config['run']['runner']['train']['criteria']
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
        'full_template_inputs': {task: bool(config['run']['data'][task]['transform'].get(
            'with_full_template_image', False)) for task in ('test', 'eval')},
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
            'ctr_parameters_sha256': fingerprint(ctr) if ctr else None}


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


def resolve_model(config_type):
    registry = dispatch_table(ROOT / 'trackit/models/methods/builder.py', {"config['type']"})
    builder_module = registry[config_type]
    builder_file = module_file(builder_module)
    classes = returned_classes(builder_file, builder_module.rpartition('.')[0])
    chains = {name: class_bases(module, name) for name, module in classes.items()}
    return {'builder': builder_module, 'classes': sorted(classes), 'bases': chains,
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
                      'model': model, 'pipeline': pipeline}
    return rows


def problems(rows):
    found = []
    for name, row in rows.items():
        if row['method'] != row['config_type']:
            found.append(f'{name}: script method {row["method"]} but config type {row["config_type"]}')
        if row['script_use_ltcp'] != row['ltcp']:
            found.append(f'{name}: script use_ltcp={row["script_use_ltcp"]} but config LTCP={row["ltcp"]}')
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
    if spm_names != {'spmtrack'}:
        found.append(f'The separate SPMTrack comparison must be only spmtrack, got {sorted(spm_names)}')
    for name in ('baseline', 'stcm_base'):
        if name not in rows or rows[name]['method'] != 'STCMTrack':
            found.append(f'{name} must exist and build STCMTrack')
        elif (rows[name]['ltcp'], rows[name]['mcc'], rows[name]['rgtc']) != (False, False, False):
            found.append(f'{name} must disable all three components')
    controlled = [rows[name] for name in COMPONENTS if name in rows]
    if len(controlled) != len(COMPONENTS):
        found.append('All eight module combinations must exist')
    if controlled:
        reference = controlled[0]
        for name in COMPONENTS:
            if name not in rows:
                continue
            row = rows[name]
            if row['method'] != 'STCMTrack' or row['model'] != reference['model'] \
                    or row['pipeline'] != reference['pipeline']:
                found.append(f'{name}: all eight ablations must use the same STCMTrack model and pipeline')
            for key in ('shared_settings_sha256', 'eval_shared_settings_sha256'):
                if row[key] != reference[key]:
                    found.append(f'{name}: non-component settings differ from baseline ({key})')
            needs_full_frame = row['mcc'] or row['rgtc']
            if any(value != needs_full_frame for value in row['full_template_inputs'].values()):
                found.append(f'{name}: full-template-image inputs must be enabled exactly when MCC or RGTC is used')
        for key in ('ltcp_parameters_sha256', 'ctr_parameters_sha256'):
            if len({row[key] for row in controlled if row[key] is not None}) > 1:
                found.append(f'Ablations use different component parameters ({key})')
        if 'stcm_base' in rows and any(rows['stcm_base'][key] != reference[key] for key in (
                'shared_settings_sha256', 'eval_shared_settings_sha256')):
            found.append('stcm_base must be an alias of baseline with identical settings')
    return found


def markdown(rows):
    out = ['| VARIANT | method | model class (train / inference) | LTCP | MCC | RGTC | train templates / search frames | '
           'post-process | eval pipeline | LTCP/CTR code reachable |',
           '|---|---|---|:---:|:---:|:---:|---|---|---|---|']
    mark = lambda value: '✓' if value else '×'
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
    return '\n'.join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', default='antiuav410', choices=('antiuav410', 'antiuav300'))
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    rows = compute(args.dataset)
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        print(markdown(rows))
    if args.check:
        found = problems(rows)
        for line in found:
            print('MISMATCH:', line, file=sys.stderr)
        if found:
            raise SystemExit(1)
        print(f'\nvariant mapping check passed for {len(rows)} variant names ({args.dataset})', file=sys.stderr)


if __name__ == '__main__':
    main()
