#!/usr/bin/env python3
"""Export a matching base/LTCP weight pair from a full training checkpoint.

Checkpoints and exported .bin files are Safetensors, including training
snapshots named model.bin. Use stcmtrack_base.bin and stcmtrack_ltcp.bin
for the default Anti-UAV410 evaluation filenames.
Exporting a stage-2 snapshot also exports its frozen base so the two files
cannot accidentally refer to different stages/datasets when used as a pair.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from safetensors.torch import load_file, save_file

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trackit.core.utils.stcmtrack_weights import validate_query_numerics

METADATA_KEYS = ('_expert_alpha', '_use_rsexpert')


def export_weights(checkpoint, base_output, ltcp_output=None):
    checkpoint, base_output = Path(checkpoint), Path(base_output)
    ltcp_output = Path(ltcp_output) if ltcp_output is not None else None
    outputs = [base_output] + ([ltcp_output] if ltcp_output is not None else [])
    manifest_output = base_output.with_suffix('.manifest.json')
    if len({p.resolve() for p in [checkpoint, *outputs, manifest_output]}) != len(outputs) + 2:
        raise ValueError('Input, output, and manifest paths must be distinct')
    if any(p.exists() for p in outputs) or base_output.with_suffix('.manifest.json').exists():
        raise FileExistsError('An output already exists; choose new output paths')
    # Top-level entries whose names start with an underscore are metadata; the TMoE scaling is the only one exported.
    state = {k: v for k, v in load_file(str(checkpoint), device='cpu').items()
             if not k.startswith('_') or k in METADATA_KEYS}
    required = {*METADATA_KEYS, 'patch_embed.proj.weight', 'pos_embed', 'track_query', 'query_embed',
                'token_type_embed', 'head.cls_mlp.layers.0.weight', 'head.reg_mlp.layers.0.weight'}
    if not required <= set(state) or not any(k.startswith('blocks.') for k in state):
        raise ValueError('Input is not a full tracking-network checkpoint written by train_stcmtrack.sh')
    ltcp_keys = {k for k in state if k.startswith('ltcp.')}
    if ltcp_output is not None and ltcp_keys != {'ltcp.gate.weight', 'ltcp.gate.bias'}:
        raise ValueError('LTCP export requires a stage-2 checkpoint containing the complete gate')
    if ltcp_keys and ltcp_output is None:
        raise ValueError('Supply --ltcp-output to preserve the LTCP weights in this checkpoint')
    validate_query_numerics(state['track_query'].detach().cpu().double().numpy(),
                            state['query_embed'].detach().cpu().double().numpy())
    source_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    metadata = {'source_checkpoint_sha256': source_sha256}
    base_output.parent.mkdir(parents=True, exist_ok=True)
    base = {k: v.contiguous() for k, v in state.items() if not k.startswith('ltcp.')}
    save_file(base, str(base_output), metadata={**metadata, 'component': 'base'})
    if ltcp_output is not None:
        ltcp_output.parent.mkdir(parents=True, exist_ok=True)
        adapter = {k: v.contiguous() for k, v in state.items() if k.startswith('ltcp.') or k in METADATA_KEYS}
        save_file(adapter, str(ltcp_output), metadata={**metadata, 'component': 'ltcp'})
    manifest = {**metadata, 'source': str(checkpoint),
                'outputs': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs}}
    base_output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--base-output', type=Path, required=True,
                        help='Base Safetensors output, e.g. weights/stcmtrack_base.bin')
    parser.add_argument('--ltcp-output', type=Path,
                        help='LTCP Safetensors output, e.g. weights/stcmtrack_ltcp.bin')
    args = parser.parse_args()
    print(json.dumps(export_weights(args.checkpoint, args.base_output, args.ltcp_output), indent=2))


if __name__ == '__main__':
    main()
