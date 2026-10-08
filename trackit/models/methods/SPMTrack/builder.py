"""Model builder for the SPMTrack baseline (pinned to WenRuiCai/SPMTrack @ c581fe27231f3e16c38578e47daddadfaf6ffd7d).

Structure follows upstream trackit/models/methods/SPMTrack/builder.py. The training class and the
streaming inference class are built from the same constructor arguments.
"""
from trackit.models import ModelBuildingContext, ModelImplSuggestions
from trackit.models.backbone.builder import build_backbone
from trackit.miscellanies.pretty_format import pretty_format
from .sample_data_generator import build_sample_input_data_generator

SPMTRACK_BUILD_STRING = 'SPMTrack_c581fe2_port_v1'


def get_SPMTrack_build_context(config: dict):
    print('SPMTrack model config:\n' + pretty_format(config['model']))
    return ModelBuildingContext(lambda impl_advice: build_SPMTrack_model(config, impl_advice),
                                lambda impl_advice: get_SPMTrack_build_string(config['model'], impl_advice),
                                build_sample_input_data_generator(config))


def build_SPMTrack_model(config: dict, model_impl_suggestions: ModelImplSuggestions):
    model_config = config['model']
    common_config = config['common']
    model_type = model_config['type']
    if model_type != 'dinov2':
        # `dinov2_full_finetune` exists in the upstream builder but cannot be constructed there:
        # it calls the TMoE class without the TMoE arguments.
        raise NotImplementedError(f"SPMTrack model type '{model_type}' is not supported (only 'dinov2').")
    ltcp_config = model_config.get('ltcp', {})
    if ltcp_config.get('enabled', False):
        raise ValueError('SPMTrack is the baseline and has no LTCP; remove model.ltcp (use STCMTrack for LTCP).')
    backbone = build_backbone(model_config['backbone'],
                              torch_jit_trace_compatible=model_impl_suggestions.torch_jit_trace_compatible)
    tmoe_config = model_config['tmoe']
    model_args = (backbone, common_config['template_feat_size'], common_config['search_region_feat_size'],
                  tmoe_config['r'], tmoe_config['alpha'], tmoe_config['dropout'], tmoe_config['use_rsexpert'],
                  tmoe_config['expert_nums'], tmoe_config['init_method'])
    model_kwargs = dict(shared_expert=tmoe_config['shared_expert'], route_compression=tmoe_config['route_compression'],
                        allow_unmarked_weights=model_config.get('allow_unmarked_weights', False))
    if model_impl_suggestions.optimize_for_inference:
        from .SPMTrack_inference import SPMTrackInference_DINOv2
        return SPMTrackInference_DINOv2(*model_args, **model_kwargs)
    from .SPMTrack import SPMTrack_DINOv2
    return SPMTrack_DINOv2(*model_args, **model_kwargs)


def get_SPMTrack_build_string(model_config: dict, model_impl_suggestions: ModelImplSuggestions):
    build_string = SPMTRACK_BUILD_STRING
    if model_impl_suggestions.optimize_for_inference:
        build_string += '_inference'
    if model_impl_suggestions.torch_jit_trace_compatible:
        build_string += '_disable_flash_attn'
    return build_string
