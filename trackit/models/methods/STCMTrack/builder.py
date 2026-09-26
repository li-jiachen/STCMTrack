from trackit.models import ModelBuildingContext, ModelImplSuggestions
from trackit.models.backbone.builder import build_backbone
from trackit.miscellanies.pretty_format import pretty_format
from .sample_data_generator import build_sample_input_data_generator


def get_STCMTrack_build_context(config: dict):
    print('STCMTrack model config:\n' + pretty_format(config['model']))
    return ModelBuildingContext(lambda impl_advice: build_STCMTrack_model(config, impl_advice),
                                lambda impl_advice: get_STCMTrack_build_string(config['model'], impl_advice),
                                build_sample_input_data_generator(config))


def build_STCMTrack_model(config: dict, model_impl_suggestions: ModelImplSuggestions):
    model_config = config['model']
    common_config = config['common']
    ltcp_config = model_config.get('ltcp', {})
    backbone = build_backbone(model_config['backbone'],
                              torch_jit_trace_compatible=model_impl_suggestions.torch_jit_trace_compatible)
    model_type = model_config['type']
    if model_type != 'dinov2':
        raise NotImplementedError(f"Model type '{model_type}' is not supported.")
    tmoe_config = model_config['tmoe']
    model_args = (backbone, common_config['template_feat_size'], common_config['search_region_feat_size'],
                  tmoe_config['r'], tmoe_config['alpha'], tmoe_config['dropout'], tmoe_config['use_rsexpert'],
                  tmoe_config['expert_nums'], tmoe_config['init_method'])
    model_kwargs = dict(shared_expert=tmoe_config['shared_expert'], route_compression=tmoe_config['route_compression'],
                        ltcp_config=ltcp_config)
    if model_impl_suggestions.optimize_for_inference:
        from .STCMTrack_inference import STCMTrackInference_DINOv2
        model = STCMTrackInference_DINOv2(*model_args, **model_kwargs)
    else:
        from .STCMTrack import STCMTrack_DINOv2
        model = STCMTrack_DINOv2(*model_args, **model_kwargs)
    return model


def get_STCMTrack_build_string(model_config: dict, model_impl_suggestions: ModelImplSuggestions):
    ltcp_config = model_config.get('ltcp', {})
    ltcp_enabled = ltcp_config.get('enabled', False)
    ltcp_train_only = ltcp_config.get('train_only', False)
    build_string = 'STCMTrack'
    if model_impl_suggestions.optimize_for_inference:
        build_string += '_merged'
    if ltcp_enabled:
        build_string += '_ltcp'
        if ltcp_train_only:
            build_string += '_train_only'
    if model_impl_suggestions.torch_jit_trace_compatible:
        build_string += '_disable_flash_attn'
    return build_string


