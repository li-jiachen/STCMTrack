
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import torch
from safetensors.torch import save_file

from trackit.core.utils.stcmtrack_weights import PUBLISHED_WEIGHT_SHA256, file_sha256
from trackit.models.backbone.dinov2 import DinoVisionTransformer
from trackit.models.backbone.dinov2.builder import build_dino_v2_backbone
from trackit.models.methods.STCMTrack.STCMTrack import STCMTrack_DINOv2
from tools.check_stcmtrack_weights import validate_weight_pair


torch.set_num_threads(2)


def make_model(pretrained=True, stage2=True, ltcp=True):
    vit = DinoVisionTransformer(img_size=28, patch_size=14, embed_dim=32, depth=1,
                               num_heads=4, block_chunks=0, init_values=1e-5)
    
    vit._pretrained_weights_loaded = pretrained
    return STCMTrack_DINOv2(vit, (1, 1), (2, 2), 4, 4., 0., expert_nums=2,
                           ltcp_config={'enabled': ltcp, 'train_only': stage2, 'print_summary': False})


def published_states(model):
    state = model.state_dict()
    scaling = {'expert_alpha': state['_expert_alpha'].clone(),
               'use_rsexpert': state['_use_rsexpert'].clone()}
    base = {key: state[key].clone() for key in model._tracking_trainable_keys}
    gate = {key: value.clone() for key, value in state.items() if key.startswith('ltcp.')}
    return {**base, **scaling}, {**gate, **scaling}


class PublishedCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base_path = Path(self.directory.name) / 'base.bin'
        self.gate_path = Path(self.directory.name) / 'ltcp.bin'
        self.source = make_model()
        self.base, self.gate = published_states(self.source)

    def write_pair(self, base=None, gate=None):
        save_file(self.base if base is None else base, str(self.base_path))
        save_file(self.gate if gate is None else gate, str(self.gate_path))
        
        
        identities = {'base': file_sha256(self.base_path), 'ltcp': file_sha256(self.gate_path)}
        context = patch.dict(PUBLISHED_WEIGHT_SHA256, identities)
        context.start()
        self.addCleanup(context.stop)

    def assert_unchanged(self, before, model):
        for key, value in before.items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)

    def test_loads_all_stage1_parameters_after_stage2_freeze_and_preserves_backbone(self):
        self.write_pair()
        target = make_model()
        self.assertEqual([key for key, parameter in target.named_parameters() if parameter.requires_grad],
                         ['ltcp.gate.weight', 'ltcp.gate.bias'])
        frozen = {key: value.clone() for key, value in target.state_dict().items()
                  if key not in target._tracking_trainable_keys and not key.startswith('ltcp.')}
        result = target.load_checkpoint_file(self.base_path)
        self.assertFalse(result.missing_keys or result.unexpected_keys)
        target.load_checkpoint_file(self.gate_path)
        for key in target._tracking_trainable_keys | {'ltcp.gate.weight', 'ltcp.gate.bias'}:
            torch.testing.assert_close(target.state_dict()[key], self.source.state_dict()[key], rtol=0, atol=0)
        for key, value in frozen.items():
            torch.testing.assert_close(target.state_dict()[key], value, rtol=0, atol=0)
        checked = validate_weight_pair(self.base_path, self.gate_path)
        self.assertEqual(checked['verification'], 'published_file_sha256')
        self.assertIsNone(checked['source_checkpoint_sha256'])

    def test_single_base_for_stage1_checkpoint(self):
        self.write_pair()
        target = make_model(ltcp=False)
        target.load_checkpoint_file(self.base_path)
        self.assertTrue(target._base_checkpoint_loaded)
        self.assertEqual(validate_weight_pair(self.base_path)['status'], 'passed')

    def test_gate_requires_matching_published_base(self):
        self.write_pair()
        target = make_model()
        before = {key: value.clone() for key, value in target.state_dict().items()}
        with self.assertRaisesRegex(ValueError, 'published base before'):
            target.load_checkpoint_file(self.gate_path)
        self.assert_unchanged(before, target)
        target.load_state_dict(self.source.state_dict())
        with self.assertRaisesRegex(ValueError, 'published base before'):
            target.load_checkpoint_file(self.gate_path)

    def test_unrecognized_file_and_mapping_without_authentication_are_rejected(self):
        save_file(self.base, str(self.base_path))
        target = make_model()
        with self.assertRaisesRegex(ValueError, 'Unrecognized published'):
            target.load_checkpoint_file(self.base_path)
        with self.assertRaisesRegex(ValueError, 'Unrecognized published'):
            validate_weight_pair(self.base_path)
        with self.assertRaisesRegex(ValueError, 'authenticated'):
            target.load_state_dict(self.base, strict=False)

    def test_random_backbone_is_rejected_before_copy(self):
        self.write_pair()
        target = make_model(pretrained=False)
        before = {key: value.clone() for key, value in target.state_dict().items()}
        with self.assertRaisesRegex(ValueError, 'pretrained DINOv2'):
            target.load_checkpoint_file(self.base_path)
        self.assert_unchanged(before, target)

    def test_incomplete_or_unknown_learned_keys_are_rejected_before_copy(self):
        for kind in ('missing', 'unknown'):
            with self.subTest(kind=kind):
                changed = dict(self.base)
                if kind == 'missing':
                    changed.pop('track_query')
                else:
                    changed['unknown.weight'] = torch.ones(1)
                self.write_pair(base=changed)
                target = make_model()
                before = {key: value.clone() for key, value in target.state_dict().items()}
                with self.assertRaisesRegex(ValueError, 'published base|Unexpected'):
                    target.load_checkpoint_file(self.base_path)
                self.assert_unchanged(before, target)

    def test_shapes_and_scaling_are_rejected_before_copy(self):
        for key in ('head.cls_mlp.layers.0.weight', 'expert_alpha'):
            with self.subTest(key=key):
                changed = dict(self.base)
                changed[key] = torch.ones(1) if key.startswith('head.') else torch.tensor(8.)
                self.write_pair(base=changed)
                target = make_model()
                before = {name: value.clone() for name, value in target.state_dict().items()}
                with self.assertRaisesRegex(ValueError, 'shape mismatch|configuration mismatch'):
                    target.load_checkpoint_file(self.base_path)
                self.assert_unchanged(before, target)

    def test_partial_or_wrong_shape_gate_is_rejected_before_copy(self):
        for kind in ('missing', 'shape'):
            with self.subTest(kind=kind):
                gate = dict(self.gate)
                if kind == 'missing':
                    gate.pop('ltcp.gate.bias')
                else:
                    gate['ltcp.gate.weight'] = torch.ones(1, 3)
                self.write_pair(gate=gate)
                target = make_model()
                target.load_checkpoint_file(self.base_path)
                before = {key: value.clone() for key, value in target.state_dict().items()}
                with self.assertRaisesRegex(ValueError, 'does not match|shape mismatch'):
                    target.load_checkpoint_file(self.gate_path)
                self.assert_unchanged(before, target)

    def test_mixed_formats_are_rejected(self):
        self.write_pair()
        target = make_model()
        target.load_checkpoint_file(self.base_path)
        with self.assertRaisesRegex(ValueError, 'Cannot mix'):
            target.load_state_dict(self.source.export_ltcp_state_dict(), strict=False)
        save_file(self.source.export_ltcp_state_dict(), str(self.gate_path))
        with self.assertRaisesRegex(ValueError, 'Cannot mix'):
            validate_weight_pair(self.base_path, self.gate_path)
        mixed = {**self.base, '_expert_alpha': torch.tensor(4.)}
        save_file(mixed, str(self.base_path))
        with self.assertRaisesRegex(ValueError, 'Mixed or incomplete'):
            target.load_checkpoint_file(self.base_path)

    def test_full_unknown_keys_and_shapes_are_rejected_before_copy(self):
        for kind in ('unknown', 'shape'):
            with self.subTest(kind=kind):
                state = dict(self.source.state_dict())
                if kind == 'unknown':
                    state['unknown.weight'] = torch.ones(1)
                else:
                    state['ltcp.gate.weight'] = torch.ones(1, 3)
                target = make_model()
                before = {key: value.clone() for key, value in target.state_dict().items()}
                with self.assertRaisesRegex(ValueError, 'Unexpected|shape mismatch'):
                    target.load_state_dict(state, strict=False)
                self.assert_unchanged(before, target)

    def test_authenticated_query_is_loaded_without_changing_values(self):
        changed = dict(self.base)
        changed['track_query'] = torch.zeros_like(changed['track_query'])
        changed['track_query'][0, 0] = 1.e28
        self.write_pair(base=changed)
        target = make_model()
        frozen = {key: value.clone() for key, value in target.state_dict().items()
                  if key not in target._tracking_trainable_keys and not key.startswith('ltcp.')}
        original_hashes = (file_sha256(self.base_path), file_sha256(self.gate_path))
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter('always')
            result = target.load_checkpoint_file(self.base_path, strict=True)
        self.assertEqual(seen, [])
        self.assertFalse(result.missing_keys or result.unexpected_keys)
        target.load_checkpoint_file(self.gate_path)
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter('always')
            target.load_checkpoint_file(self.base_path)
        self.assertEqual(seen, [])
        for key in target._tracking_trainable_keys:
            torch.testing.assert_close(target.state_dict()[key], changed[key], rtol=0, atol=0)
        for key in ('ltcp.gate.weight', 'ltcp.gate.bias'):
            torch.testing.assert_close(target.state_dict()[key], self.gate[key], rtol=0, atol=0)
        for key, value in frozen.items():
            torch.testing.assert_close(target.state_dict()[key], value, rtol=0, atol=0)
        self.assertEqual(target._base_checkpoint_format, 'published')
        self.assertEqual(target._published_base_sha256, original_hashes[0])
        self.assertEqual((file_sha256(self.base_path), file_sha256(self.gate_path)), original_hashes)

        without_ltcp = make_model(ltcp=False)
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter('always')
            without_ltcp.load_checkpoint_file(self.base_path)
        self.assertEqual(seen, [])
        torch.testing.assert_close(without_ltcp.track_query, changed['track_query'], rtol=0, atol=0)

    def test_authenticated_base_does_not_relax_full_or_unauthenticated_reloads(self):
        changed = dict(self.base)
        changed['track_query'] = torch.zeros_like(changed['track_query'])
        changed['track_query'][0, 0] = 1.e28
        self.write_pair(base=changed)
        target = make_model()
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter('always')
            target.load_checkpoint_file(self.base_path)
        self.assertEqual(seen, [])
        before = {key: value.clone() for key, value in target.state_dict().items()}
        load_flags = (target._base_checkpoint_loaded, target._base_checkpoint_format,
                      target._published_base_sha256)
        full = dict(self.source.state_dict())
        full['track_query'] = changed['track_query']
        full_path = Path(self.directory.name) / 'full.bin'
        save_file(full, str(full_path))
        for strict in (False, True):
            for mode in ('mapping', 'file'):
                with self.subTest(strict=strict, mode=mode):
                    with self.assertRaisesRegex(ValueError, 'overflow FP32 LayerNorm'):
                        if mode == 'mapping':
                            target.load_state_dict(full, strict=strict)
                        else:
                            target.load_checkpoint_file(full_path, strict=strict)
                    self.assert_unchanged(before, target)
                    self.assertEqual((target._base_checkpoint_loaded, target._base_checkpoint_format,
                                      target._published_base_sha256), load_flags)
            with self.assertRaisesRegex(ValueError, 'authenticated'):
                target.load_state_dict(changed, strict=strict)
            self.assert_unchanged(before, target)
            self.assertEqual((target._base_checkpoint_loaded, target._base_checkpoint_format,
                              target._published_base_sha256), load_flags)

        modified = dict(changed)
        modified['track_query'] = changed['track_query'].clone()
        modified['track_query'][0, 1] = 2.e28
        save_file(modified, str(self.base_path))
        with self.assertRaisesRegex(ValueError, 'Unrecognized published'):
            target.load_checkpoint_file(self.base_path)
        self.assert_unchanged(before, target)
        self.assertEqual((target._base_checkpoint_loaded, target._base_checkpoint_format,
                          target._published_base_sha256), load_flags)

    def test_authenticated_nonfinite_query_is_rejected_without_modifying_model(self):
        for kind in ('nan', 'inf', 'finite_sum_overflow'):
            with self.subTest(kind=kind):
                changed = dict(self.base)
                changed['track_query'] = torch.zeros_like(changed['track_query'])
                changed['query_embed'] = torch.zeros_like(changed['query_embed'])
                if kind == 'finite_sum_overflow':
                    changed['track_query'][0, 0] = 3.e38
                    changed['query_embed'][0, 0] = 3.e38
                else:
                    changed['track_query'][0, 0] = float(kind)
                self.write_pair(base=changed)
                target = make_model()
                before = {key: value.clone() for key, value in target.state_dict().items()}
                with self.assertRaisesRegex(ValueError, 'Non-finite track_query/query_embed'):
                    target.load_checkpoint_file(self.base_path)
                self.assert_unchanged(before, target)
                self.assertFalse(target._base_checkpoint_loaded)
                self.assertIsNone(target._base_checkpoint_format)
                self.assertIsNone(target._published_base_sha256)


class BackbonePretrainedMarkerTests(unittest.TestCase):
    def test_marker_is_set_only_after_successful_load(self):
        backbone = torch.nn.Linear(1, 1)
        with patch('trackit.models.backbone.dinov2.vit_base', return_value=backbone), \
                patch('torch.hub.load_state_dict_from_url', return_value=backbone.state_dict()):
            self.assertTrue(build_dino_v2_backbone('ViT-B/14', True)._pretrained_weights_loaded)
        random_backbone = torch.nn.Linear(1, 1)
        with patch('trackit.models.backbone.dinov2.vit_base', return_value=random_backbone):
            self.assertFalse(build_dino_v2_backbone('ViT-B/14', False)._pretrained_weights_loaded)
        failed_backbone = torch.nn.Linear(1, 1)
        with patch('trackit.models.backbone.dinov2.vit_base', return_value=failed_backbone), \
                patch('torch.hub.load_state_dict_from_url', return_value={}):
            with self.assertRaises(RuntimeError):
                build_dino_v2_backbone('ViT-B/14', True)
        self.assertFalse(getattr(failed_backbone, '_pretrained_weights_loaded', False))


if __name__ == '__main__':
    unittest.main()

