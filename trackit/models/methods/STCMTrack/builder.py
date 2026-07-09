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
    evidence_config = model_config.get('evidence', {})
    evidence_enabled = evidence_config.get('enabled', False)
    evidence_train_only = evidence_config.get('train_only', False)
    ltcp_config = model_config.get('ltcp', {})
    backbone = build_backbone(model_config['backbone'],
                              torch_jit_trace_compatible=model_impl_suggestions.torch_jit_trace_compatible)
    model_type = model_config['type']
    if model_type == 'dinov2':
        if model_impl_suggestions.optimize_for_inference:
            from .STCMTrack_full_finetune import STCMTrackBaseline_DINOv2
            model = STCMTrackBaseline_DINOv2(backbone, common_config['template_feat_size'], common_config['search_region_feat_size'],
                                         model_config['tmoe']['r'], model_config['tmoe']['alpha'],
                                         model_config['tmoe']['dropout'], model_config['tmoe']['use_rsexpert'],
                                         model_config['tmoe']['expert_nums'], model_config['tmoe']['init_method'],
                                         shared_expert=model_config['tmoe']['shared_expert'], route_compression=model_config['tmoe']['route_compression'],
                                         evidence_enabled=evidence_enabled, ltcp_config=ltcp_config)
        else:
            from .STCMTrack import STCMTrack_DINOv2
            model = STCMTrack_DINOv2(backbone, common_config['template_feat_size'], common_config['search_region_feat_size'],
                                 model_config['tmoe']['r'], model_config['tmoe']['alpha'],
                                 model_config['tmoe']['dropout'], model_config['tmoe']['use_rsexpert'],
                                 model_config['tmoe']['expert_nums'], model_config['tmoe']['init_method'],
                                 shared_expert=model_config['tmoe']['shared_expert'], route_compression=model_config['tmoe']['route_compression'],
                                 evidence_enabled=evidence_enabled, evidence_train_only=evidence_train_only, ltcp_config=ltcp_config)
    elif model_type == 'dinov2_full_finetune':
        from .STCMTrack_full_finetune import STCMTrackBaseline_DINOv2
        model = STCMTrackBaseline_DINOv2(backbone, common_config['template_feat_size'], common_config['search_region_feat_size'],
                                         model_config['tmoe']['r'], model_config['tmoe']['alpha'],
                                         model_config['tmoe']['dropout'], model_config['tmoe']['use_rsexpert'],
                                         model_config['tmoe']['expert_nums'], model_config['tmoe']['init_method'],
                                         shared_expert=model_config['tmoe']['shared_expert'], route_compression=model_config['tmoe']['route_compression'],
                                         evidence_enabled=evidence_enabled, ltcp_config=ltcp_config)
    else:
        raise NotImplementedError(f"Model type '{model_type}' is not supported.")
    return model


def get_STCMTrack_build_string(model_config: dict, model_impl_suggestions: ModelImplSuggestions):
    model_type = model_config['type']
    evidence_enabled = model_config.get('evidence', {}).get('enabled', False)
    ltcp_config = model_config.get('ltcp', {})
    ltcp_enabled = ltcp_config.get('enabled', False)
    ltcp_train_only = ltcp_config.get('train_only', False)
    build_string = 'STCMTrack'
    if 'full_finetune' in model_type:
        build_string += '_full_finetune'
    else:
        if model_impl_suggestions.optimize_for_inference:
            build_string += '_merged'
        if evidence_enabled:
            build_string += '_evidence'
        if ltcp_enabled:
            build_string += '_ltcp'
            if ltcp_train_only:
                build_string += '_train_only'
    if model_impl_suggestions.torch_jit_trace_compatible:
        build_string += '_disable_flash_attn'
    return build_string


