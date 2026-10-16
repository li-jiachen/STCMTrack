
import hashlib


SCALING_KEYS = ('_expert_alpha', '_use_rsexpert')
LEGACY_SCALING_KEYS = ('expert_alpha', 'use_rsexpert')
LTCP_KEYS = ('ltcp.gate.weight', 'ltcp.gate.bias')
PUBLISHED_LEGACY_SHA256 = {
    'base': '71eb8f20cbdec8bf2dff5177806923eca0fa08b694c2b40cf5399f2d9ad2d2d0',
    'ltcp': 'ed4e063bcac94e6fd98de14abb1416965f4a26ad92f482602234807228b291b7',
}


def checkpoint_scaling_format(keys):
    keys = set(keys)
    old, new = keys & set(LEGACY_SCALING_KEYS), keys & set(SCALING_KEYS)
    if old:
        if old != set(LEGACY_SCALING_KEYS) or new:
            raise ValueError('Mixed or incomplete legacy/modern TMoE scaling entries')
        return 'legacy'
    if new != set(SCALING_KEYS):
        raise ValueError('Not an STCMTrack checkpoint: TMoE scaling entries are missing')
    return 'modern'


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def authenticate_published_legacy(path, component):
    digest = file_sha256(path)
    if digest != PUBLISHED_LEGACY_SHA256[component]:
        raise ValueError(f'Unrecognized legacy {component} file: only the original published '
                         'Anti-UAV410 .bin files are supported')
    return digest


def validate_query_numerics(track_query, query_embed, *, published_legacy_sha256=None):
    
    import numpy as np
    with np.errstate(over='ignore', invalid='ignore'):
        query = np.asarray(track_query, dtype=np.float32) + np.asarray(query_embed, dtype=np.float32)
    if not np.isfinite(query).all():
        raise ValueError('Non-finite track_query/query_embed in checkpoint')
    variance = np.var(query.astype(np.float64), axis=-1, dtype=np.float64)
    fp32_max = np.finfo(np.float32).max
    magnitude_overflow = np.any(np.abs(query) > fp32_max)
    variance_overflow = np.any(variance > fp32_max)
    if magnitude_overflow or variance_overflow:
        diagnostic = ('track_query/query_embed overflow FP32 LayerNorm '
                      f'(combined query absmax={np.max(np.abs(query)):.6e}, '
                      f'variance={np.max(variance):.6e})')
        if (variance_overflow and not magnitude_overflow
                and published_legacy_sha256 == PUBLISHED_LEGACY_SHA256['base']):
            return
        else:
            raise ValueError(diagnostic)
