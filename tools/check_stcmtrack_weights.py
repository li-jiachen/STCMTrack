#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import sys
from safetensors import safe_open

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trackit.core.utils.stcmtrack_weights import (
    SCALING_KEYS, PUBLISHED_SCALING_KEYS, LTCP_KEYS,
    authenticate_published_weights, checkpoint_scaling_format, validate_query_numerics)


def _inspect(path, *, component):
    with safe_open(str(path), framework='np') as f:
        all_keys = set(f.keys())
        format_name = checkpoint_scaling_format(all_keys)
        published_sha256 = authenticate_published_weights(path, component) if format_name == 'published' else None
        scaling_keys = PUBLISHED_SCALING_KEYS if format_name == 'published' else SCALING_KEYS

        keys = all_keys if format_name == 'published' else {
            key for key in all_keys if not key.startswith('_') or key in SCALING_KEYS}
        if any(f.get_slice(key).get_shape() != [] for key in scaling_keys):
            raise ValueError(f'{path}: TMoE scaling entries must be scalars')
        scaling = dict(zip(SCALING_KEYS, (f.get_tensor(key).item() for key in scaling_keys)))
        if 'track_query' in keys and 'query_embed' in keys:
            validate_query_numerics(f.get_tensor('track_query'), f.get_tensor('query_embed'),
                                    published_weight_sha256=published_sha256)
        return keys, f.metadata() or {}, scaling, format_name, published_sha256


def validate_weight_pair(base_path, ltcp_path=None):
    keys, meta, scaling, format_name, base_sha256 = _inspect(base_path, component='base')
    if any(k.startswith('ltcp.') for k in keys) or not any(k.startswith('head.') for k in keys):
        raise ValueError('Base file must contain the tracking network without LTCP; '
                         'use export_stcmtrack_weights.py')
    delta_sha256 = None
    if ltcp_path is not None:
        delta_keys, delta_meta, delta_scaling, delta_format, delta_sha256 = _inspect(ltcp_path, component='ltcp')
        if delta_format != format_name:
            raise ValueError('Cannot mix published and full-checkpoint base/LTCP formats')
        expected_scaling = PUBLISHED_SCALING_KEYS if format_name == 'published' else SCALING_KEYS
        if delta_keys != {*LTCP_KEYS, *expected_scaling} or delta_scaling != scaling:
            raise ValueError('LTCP file has incorrect keys or TMoE scaling')
        if format_name != 'published':
            source = meta.get('source_checkpoint_sha256')
            if (not source or source != delta_meta.get('source_checkpoint_sha256')
                    or meta.get('component') != 'base' or delta_meta.get('component') != 'ltcp'):
                raise ValueError('Base/LTCP provenance differs or is missing; export both from the SAME stage-2 checkpoint')
    return {'status': 'passed', 'base': str(base_path), 'ltcp': str(ltcp_path) if ltcp_path is not None else None,
            'format': format_name,
            'verification': 'published_file_sha256' if format_name == 'published' else 'training_snapshot_metadata',
            'base_sha256': base_sha256, 'ltcp_sha256': delta_sha256,
            'source_checkpoint_sha256': None if format_name == 'published' else meta.get('source_checkpoint_sha256')}


def main():
    parser = argparse.ArgumentParser(description='Check STCMTrack weight files.', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base', required=True, type=Path, help='Base model .bin file (Safetensors format)')
    parser.add_argument('--ltcp', type=Path, help='LTCP .bin file (Safetensors format)')
    args = parser.parse_args()
    try:
        result = validate_weight_pair(args.base, args.ltcp)
    except (ValueError, OSError) as exc:
        raise SystemExit(f'STCMTrack weight check failed: {exc}')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
