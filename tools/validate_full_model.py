#!/usr/bin/env python3
"""Build the full-size model (12-layer ViT-B/14, 196 / 378 inputs) on the CPU and check it.

The script checks the tensor shapes, the parameters that are trained in stage 2, and that
frame-by-frame inference with the LTCP memory reproduces the three-frame training forward
pass. It uses a randomly initialized network and random inputs, so nothing is downloaded.
"""
import argparse
import contextlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from trackit.core.boot.funcs.utils.custom_yaml_loader import load_yaml
from trackit.core.boot.funcs.mixin import apply_static_mixin_rules
from trackit.models import ModelImplSuggestions
from trackit.models.methods.STCMTrack.builder import build_STCMTrack_model
from trackit.models.compiling.plain import PlainWrapper


def validate():
    torch.set_num_threads(2)
    torch.manual_seed(31)
    root = Path(__file__).resolve().parents[1]
    config = load_yaml(str(root / 'config/STCMTrack/dinov2/config.yaml'))
    for name in ('ltcp', 'ltcp_stage2'):
        apply_static_mixin_rules(load_yaml(str(root / f'config/STCMTrack/_mixin/{name}.yaml')), config)
    config['model']['backbone']['parameters']['pretrained'] = False
    started = time.perf_counter()
    model = build_STCMTrack_model(config, ModelImplSuggestions(optimize_for_inference=True)).eval()
    assert model.embed_dim == 768 and len(model.blocks) == 12
    assert [n for n, p in model.named_parameters() if p.requires_grad] == ['ltcp.gate.weight', 'ltcp.gate.bias']
    z = torch.randn(1, 3, 196, 196)
    mask = torch.ones(1, 14, 14, dtype=torch.long)
    xs = [torch.randn(1, 3, 378, 378) for _ in range(3)]
    lengths = []
    handle = model.blocks[0].register_forward_pre_hook(lambda _, args: lengths.append(args[0].shape[1]))
    with torch.inference_mode():
        expected = model(z, mask, *xs)
    print('Full-shape clip forward completed.', flush=True)
    wrapper = PlainWrapper(model, contextlib.nullcontext)
    wrapper.epoch_begin(1)
    errors = []
    for i, x in enumerate(xs):
        result = wrapper({'ids': [13], 'z_0': z, 'z_0_feat_mask': mask, 'x': x})
        assert tuple(result['score_map'].shape) == (1, 27, 27)
        assert tuple(result['boxes'].shape) == (1, 27, 27, 4)
        assert len(model.ltcp_memory_dicts[13]) == min(i + 1, 2)
        for key in ('score_map', 'boxes'):
            assert torch.isfinite(result[key]).all()
            torch.testing.assert_close(result[key], expected[i][key], rtol=1e-6, atol=1e-7)
            errors.append(float((result[key] - expected[i][key]).abs().max()))
        print(f'Full-shape streaming frame {i + 1} completed.', flush=True)
    assert lengths == [926] * 6
    handle.remove()
    wrapper.epoch_end()
    assert not model.ltcp_memory_dicts
    return {'status': 'passed', 'backbone': 'DINOv2 ViT-B/14', 'layers': 12, 'embed_dim': 768,
            'template_size': [196, 196], 'search_size': [378, 378], 'joint_tokens': 926,
            'clip_frames': 3, 'streaming_frames': 3, 'max_absolute_error': max(errors),
            'parameter_count': sum(p.numel() for p in model.parameters()),
            'trainable_parameter_count': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'seconds': round(time.perf_counter() - started, 3), 'device': 'cpu'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = validate()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
