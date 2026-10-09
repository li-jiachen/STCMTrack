#!/usr/bin/env python3
"""Check the STCMTrack weight files before evaluation.

The base file must hold the tracking network without LTCP, the LTCP file exactly the gate
parameters, and both must have been exported from the same training snapshot
(tools/export_stcmtrack_weights.py records the SHA-256 of that snapshot in both files).
The .bin files use Safetensors format. Only the index and the TMoE scaling are
read; no model is built. Other filename extensions are also accepted.
"""
import argparse
import json
from pathlib import Path
from safetensors import safe_open

SCALING_KEYS = ('_expert_alpha', '_use_rsexpert')
LTCP_KEYS = ('ltcp.gate.weight', 'ltcp.gate.bias')


def _inspect(path):
    with safe_open(str(path), framework='np') as f:
        # Top-level entries whose names start with an underscore are metadata; the TMoE scaling is the only one read.
        keys = {key for key in f.keys() if not key.startswith('_') or key in SCALING_KEYS}
        if not set(SCALING_KEYS) <= keys:
            raise ValueError(f'{path}: not an STCMTrack checkpoint '
                             f'(the TMoE scaling entries _expert_alpha and _use_rsexpert are missing)')
        return keys, f.metadata() or {}, {key: f.get_tensor(key).item() for key in SCALING_KEYS}


def validate_weight_pair(base_path, ltcp_path=None):
    keys, meta, scaling = _inspect(base_path)
    if any(k.startswith('ltcp.') for k in keys) or not any(k.startswith('head.') for k in keys):
        raise ValueError('Base file must contain the tracking network without LTCP; '
                         'use export_stcmtrack_weights.py')
    if ltcp_path is not None:
        delta_keys, delta_meta, delta_scaling = _inspect(ltcp_path)
        if delta_keys != {*LTCP_KEYS, *SCALING_KEYS} or delta_scaling != scaling:
            raise ValueError('LTCP file has incorrect keys or TMoE scaling')
        source = meta.get('source_checkpoint_sha256')
        if (not source or source != delta_meta.get('source_checkpoint_sha256')
                or meta.get('component') != 'base' or delta_meta.get('component') != 'ltcp'):
            raise ValueError('Base/LTCP provenance differs or is missing; export both from the SAME stage-2 checkpoint')
    return {'status': 'passed', 'base': str(base_path), 'ltcp': str(ltcp_path) if ltcp_path is not None else None,
            'source_checkpoint_sha256': meta.get('source_checkpoint_sha256')}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
