#!/usr/bin/env python3
"""Check that a weight file is an SPMTrack baseline checkpoint before evaluation.

Reads only the safetensors index and a few scalar tensors; it builds no model.

* STCMTrack checkpoints (TMoE buffers named `_expert_alpha` / `_use_rsexpert`, or `ltcp.*` keys)
  are always rejected. They belong to a single-template model with a different prediction path
  and are never SPMTrack weights, whatever the file is called.
* A file written by this code carries `_spmtrack_port_version` = 1 and is accepted.
* A file without that marker is accepted only with `--allow-unmarked`, which declares that it
  was trained with the official SPMTrack code
  (github.com/WenRuiCai/SPMTrack @ c581fe27231f3e16c38578e47daddadfaf6ffd7d).
  Nothing is ever written to the file, and no marker is added to it.
"""
import argparse
import json
import math
from pathlib import Path
from safetensors import safe_open

PORT_VERSION_KEY = '_spmtrack_port_version'
STCMTRACK_KEYS = ('_expert_alpha', '_use_rsexpert')
REQUIRED_PREFIXES = ('head.', 'track_query', 'query_embed', 'token_type_embed')


def validate_spmtrack_weights(path, *, allow_unmarked=False, expected_alpha=64.0, expected_rsexpert=False):
    path = Path(path)
    # The numpy backend reads the index and a few scalars; it needs no torch and loads no weights.
    with safe_open(str(path), framework='np') as f:
        keys = set(f.keys())
        foreign = sorted(k for k in keys if k in STCMTRACK_KEYS or k.startswith('ltcp.'))
        if foreign:
            raise ValueError(f'{path}: STCMTrack checkpoint (found {foreign[:4]}); these are not SPMTrack weights')
        for prefix in REQUIRED_PREFIXES:
            if not any(k == prefix or k.startswith(prefix) for k in keys):
                raise ValueError(f'{path}: no {prefix!r} parameters; not an SPMTrack tracking network')
        if not any('.tmoe.' in k for k in keys):
            raise ValueError(f'{path}: no TMoE parameters; not an SPMTrack checkpoint')
        if 'expert_alpha' not in keys or 'use_rsexpert' not in keys:
            raise ValueError(f'{path}: the TMoE scaling metadata (expert_alpha / use_rsexpert) is missing')
        alpha = float(f.get_tensor('expert_alpha').item())
        rsexpert = bool(f.get_tensor('use_rsexpert').item())
        if not math.isclose(alpha, expected_alpha, rel_tol=1e-6) or rsexpert != expected_rsexpert:
            raise ValueError(f'{path}: TMoE scaling alpha={alpha}, use_rsexpert={rsexpert} differs from the '
                             f'configured alpha={expected_alpha}, use_rsexpert={expected_rsexpert}')
        if PORT_VERSION_KEY in keys:
            version = int(f.get_tensor(PORT_VERSION_KEY).item())
            if version != 1:
                raise ValueError(f'{path}: unsupported {PORT_VERSION_KEY}={version}')
            provenance = 'written by this code (marker present)'
        elif allow_unmarked:
            provenance = 'no marker; accepted as official SPMTrack weights (--allow-unmarked)'
        else:
            raise ValueError(f'{path}: no {PORT_VERSION_KEY} marker. Pass --allow-unmarked '
                             f'(test_stcmtrack.sh: ALLOW_UNMARKED_SPMTRACK_WEIGHTS=1) for a file that was trained '
                             f'with the official SPMTrack code.')
    return {'status': 'passed', 'weights': str(path), 'tensors': len(keys), 'provenance': provenance,
            'expert_alpha': alpha, 'use_rsexpert': rsexpert}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--weights', required=True, type=Path)
    parser.add_argument('--allow-unmarked', action='store_true')
    parser.add_argument('--expected-alpha', type=float, default=64.0)
    args = parser.parse_args()
    try:
        print(json.dumps(validate_spmtrack_weights(args.weights, allow_unmarked=args.allow_unmarked,
                                                   expected_alpha=args.expected_alpha), indent=2))
    except (ValueError, OSError) as exc:
        raise SystemExit(f'SPMTrack weight check failed: {exc}')


if __name__ == '__main__':
    main()
