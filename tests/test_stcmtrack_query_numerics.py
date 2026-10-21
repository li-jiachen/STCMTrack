import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import uuid
import warnings

import numpy as np

from trackit.core.utils.stcmtrack_weights import (
    PUBLISHED_WEIGHT_SHA256, file_sha256, validate_query_numerics)


class PublishedQueryNumericsTests(unittest.TestCase):
    def setUp(self):
        self.query = np.array([[1.e28, -1.e28, 1.e28, -1.e28]], dtype=np.float32)
        self.embedding = np.zeros_like(self.query)

    def test_normal_queries_are_accepted_without_changing_values(self):
        query = np.array([[.01, -.02, .03, -.04]], dtype=np.float64)
        embedding = np.ones_like(query) * .02
        before = query.tobytes(), embedding.tobytes()
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter('always')
            result = validate_query_numerics(query, embedding)
        self.assertIsNone(result)
        self.assertEqual(seen, [])
        self.assertEqual((query.tobytes(), embedding.tobytes()), before)

    def test_unauthenticated_variance_overflow_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'overflow FP32 LayerNorm'):
            validate_query_numerics(self.query, self.embedding)

    def test_wrong_and_ltcp_identities_do_not_allow_base_exception(self):
        for identity in ('0' * 64, PUBLISHED_WEIGHT_SHA256['ltcp']):
            with self.subTest(identity=identity):
                with self.assertRaisesRegex(ValueError, 'overflow FP32 LayerNorm'):
                    validate_query_numerics(self.query, self.embedding,
                                            published_weight_sha256=identity)

    def test_authenticated_base_is_accepted_without_changing_values(self):
        before = self.query.tobytes(), self.embedding.tobytes()
        identity = PUBLISHED_WEIGHT_SHA256['base']
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter('always')
            result = validate_query_numerics(self.query, self.embedding,
                                            published_weight_sha256=identity)
        self.assertIsNone(result)
        self.assertEqual(seen, [])
        self.assertEqual((self.query.tobytes(), self.embedding.tobytes()), before)

    def test_nonfinite_values_are_rejected_even_for_authenticated_base(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                query = self.query.copy()
                query[0, 0] = value
                with warnings.catch_warnings(record=True) as seen:
                    warnings.simplefilter('always')
                    with self.assertRaisesRegex(ValueError, 'Non-finite'):
                        validate_query_numerics(query, self.embedding,
                                                published_weight_sha256=PUBLISHED_WEIGHT_SHA256['base'])
                self.assertEqual(seen, [])

    def test_fp32_addition_overflow_cannot_use_exception(self):
        query = np.full((1, 4), np.finfo(np.float32).max, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            validate_query_numerics(query, query,
                                    published_weight_sha256=PUBLISHED_WEIGHT_SHA256['base'])

    def test_cast_overflow_cannot_use_exception(self):
        query = np.full((1, 4), 1.e50, dtype=np.float64)
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            validate_query_numerics(query, np.zeros_like(query),
                                    published_weight_sha256=PUBLISHED_WEIGHT_SHA256['base'])


class PublishedQueryCheckerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from safetensors.numpy import save_file
            from tools import check_stcmtrack_weights
        except ImportError as exc:
            raise unittest.SkipTest(f'Checker tests require Safetensors, without PyTorch: {exc}')
        cls.save_file = staticmethod(save_file)
        cls.checker = check_stcmtrack_weights

    def setUp(self):
        self.temp_root = Path(tempfile.gettempdir()).resolve()
        self.directory = self.temp_root / ('stcmtrack-query-checker-' + uuid.uuid4().hex)
        self.directory.mkdir()
        self.addCleanup(self.remove_directory)
        self.base = self.directory / 'base.bin'
        self.ltcp = self.directory / 'ltcp.bin'

    def remove_directory(self):
        target = self.directory.resolve()
        if target.parent != self.temp_root or not target.name.startswith('stcmtrack-query-checker-'):
            raise RuntimeError(f'Unexpected temporary test directory: {target}')
        shutil.rmtree(target)

    def write_base(self, *, published=True, overflow=True, nonfinite=False, metadata=None):
        query = np.array([[1.e28, -1.e28, 1.e28, -1.e28]], dtype=np.float32)
        if not overflow:
            query *= 0.
        if nonfinite:
            query[0, 0] = np.nan
        alpha, rs = ('expert_alpha', 'use_rsexpert') if published else ('_expert_alpha', '_use_rsexpert')
        self.save_file({'track_query': query, 'query_embed': np.zeros_like(query),
                        'head.weight': np.ones((1, 4), dtype=np.float32),
                        alpha: np.array(4., dtype=np.float32), rs: np.array(False)},
                       str(self.base), metadata=metadata)
        return file_sha256(self.base)

    def write_ltcp(self, *, published=True, metadata=None):
        alpha, rs = ('expert_alpha', 'use_rsexpert') if published else ('_expert_alpha', '_use_rsexpert')
        self.save_file({'ltcp.gate.weight': np.ones((1, 5), dtype=np.float32),
                        'ltcp.gate.bias': np.zeros(1, dtype=np.float32),
                        alpha: np.array(4., dtype=np.float32), rs: np.array(False)},
                       str(self.ltcp), metadata=metadata)
        return file_sha256(self.ltcp)

    def test_published_identity_is_checked_before_query_numerics(self):
        self.write_base()
        with patch.object(self.checker, 'validate_query_numerics') as numeric_check:
            with self.assertRaisesRegex(ValueError, 'Unrecognized published base'):
                self.checker.validate_weight_pair(self.base)
        numeric_check.assert_not_called()

    def test_authenticated_base_and_pair_pass_with_same_hash(self):
        base_sha, ltcp_sha = self.write_base(), self.write_ltcp()
        with patch.dict(PUBLISHED_WEIGHT_SHA256, {'base': base_sha, 'ltcp': ltcp_sha}):
            for ltcp_path in (None, self.ltcp):
                with self.subTest(ltcp_path=ltcp_path):
                    with warnings.catch_warnings(record=True) as seen:
                        warnings.simplefilter('always')
                        result = self.checker.validate_weight_pair(self.base, ltcp_path)
                    self.assertEqual(seen, [])
                    self.assertEqual(result['status'], 'passed')
                    self.assertEqual(result['format'], 'published')
                    self.assertEqual(result['verification'], 'published_file_sha256')
                    self.assertEqual(result['base_sha256'], base_sha)
                    self.assertEqual(result['ltcp_sha256'], ltcp_sha if ltcp_path else None)
                    self.assertIsNone(result['source_checkpoint_sha256'])
        self.assertEqual(file_sha256(self.base), base_sha)
        self.assertEqual(file_sha256(self.ltcp), ltcp_sha)

    def test_full_overflow_rejected_even_if_digest_matches_published_whitelist(self):
        digest = self.write_base(published=False)
        with patch.dict(PUBLISHED_WEIGHT_SHA256, {'base': digest}):
            with self.assertRaisesRegex(ValueError, 'overflow FP32 LayerNorm'):
                self.checker.validate_weight_pair(self.base)

    def test_authenticated_nonfinite_base_is_rejected(self):
        digest = self.write_base(nonfinite=True)
        with patch.dict(PUBLISHED_WEIGHT_SHA256, {'base': digest}):
            with self.assertRaisesRegex(ValueError, 'Non-finite'):
                self.checker.validate_weight_pair(self.base)

    def test_unknown_published_ltcp_is_rejected(self):
        digest = self.write_base(overflow=False)
        self.write_ltcp()
        with patch.dict(PUBLISHED_WEIGHT_SHA256, {'base': digest}):
            with self.assertRaisesRegex(ValueError, 'Unrecognized published ltcp'):
                self.checker.validate_weight_pair(self.base, self.ltcp)

    def test_full_pair_requires_matching_provenance(self):
        self.write_base(published=False, overflow=False,
                        metadata={'component': 'base', 'source_checkpoint_sha256': 'stage2-a'})
        self.write_ltcp(published=False,
                        metadata={'component': 'ltcp', 'source_checkpoint_sha256': 'stage2-b'})
        with self.assertRaisesRegex(ValueError, 'provenance differs or is missing'):
            self.checker.validate_weight_pair(self.base, self.ltcp)

    def test_main_emits_valid_json_with_empty_stderr(self):
        base_sha, ltcp_sha = self.write_base(), self.write_ltcp()
        output, error = io.StringIO(), io.StringIO()
        with patch.dict(PUBLISHED_WEIGHT_SHA256, {'base': base_sha, 'ltcp': ltcp_sha}), \
                patch.object(sys, 'argv', ['check_stcmtrack_weights.py', '--base', str(self.base), '--ltcp', str(self.ltcp)]), \
                contextlib.redirect_stdout(output), contextlib.redirect_stderr(error), \
                warnings.catch_warnings():
            warnings.simplefilter('always')
            self.checker.main()
        self.assertEqual(json.loads(output.getvalue())['status'], 'passed')
        self.assertEqual(error.getvalue(), '')
        self.assertEqual(file_sha256(self.base), base_sha)

    def test_main_unknown_published_overflow_reports_identity_failure(self):
        self.write_base()
        with patch.object(sys, 'argv', ['check_stcmtrack_weights.py', '--base', str(self.base)]):
            with self.assertRaisesRegex(SystemExit, 'Unrecognized published base'):
                self.checker.main()


if __name__ == '__main__':
    unittest.main()
