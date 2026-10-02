#!/usr/bin/env python3
"""Check the STCMTrack weight files before evaluation.

The base file must hold the tracking network without LTCP, the LTCP file exactly the gate
parameters, and both must have been exported from the same training snapshot
(tools/export_paper_weights.py records the SHA-256 of that snapshot in both files).
"""
import argparse
import json
from pathlib import Path
from safetensors import safe_open


def _inspect(path):
    with safe_open(str(path), framework='pt', device='cpu') as f:
        keys = set(f.keys())
        if '_paper_implementation_version' not in keys or f.get_tensor('_paper_implementation_version').item() != 1:
            raise ValueError(f'{path}: not an STCMTrack checkpoint written by this code '
                             f'(the _paper_implementation_version marker is missing or unsupported)')
        return keys, f.metadata() or {}, {
            key: f.get_tensor(key).item() for key in ('_expert_alpha', '_use_rsexpert') if key in keys}


def validate_weight_pair(base_path, ltcp_path=None):
    keys, meta, scaling = _inspect(base_path)
    if any(k.startswith('ltcp.') for k in keys) or not any(k.startswith('head.') for k in keys):
        raise ValueError('Base file must contain the tracking network without LTCP; use export_paper_weights.py')
    if set(scaling) != {'_expert_alpha', '_use_rsexpert'}:
        raise ValueError('Missing TMoE scaling metadata in base checkpoint')
    if ltcp_path is not None:
        delta_keys, delta_meta, delta_scaling = _inspect(ltcp_path)
        expected = {'ltcp.gate.weight', 'ltcp.gate.bias', '_paper_implementation_version', '_expert_alpha', '_use_rsexpert'}
        if delta_keys != expected or delta_scaling != scaling:
            raise ValueError('LTCP file has incorrect keys or TMoE scaling')
        source = meta.get('source_checkpoint_sha256')
        if (not source or source != delta_meta.get('source_checkpoint_sha256')
                or meta.get('component') != 'base' or delta_meta.get('component') != 'ltcp'):
            raise ValueError('Base/LTCP provenance differs or is missing; export both from the SAME stage-2 checkpoint')
    return {'status': 'passed', 'base': str(base_path), 'ltcp': str(ltcp_path) if ltcp_path is not None else None,
            'source_checkpoint_sha256': meta.get('source_checkpoint_sha256')}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--ltcp', type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_weight_pair(args.base, args.ltcp), indent=2))


if __name__ == '__main__':
    main()
