import torch
from trackit.runner.evaluation.common.siamfc_search_region_cropping_params_provider.builder import \
    build_siamfc_search_region_cropping_parameter_provider_factory
from ....components.post_process.builder import build_post_process

from . import SPMTrackOneStream_Evaluation_MainPipeline


def build_spmtrack_one_stream_tracker_pipeline(pipeline_config: dict, config: dict, device: torch.device):
    common_config = config['common']

    if 'ctr' in pipeline_config:
        raise ValueError('The SPMTrack baseline has no CTR (MCC/RGTC); remove pipeline.ctr.')
    plugin_types = [plugin_config['type'] for plugin_config in pipeline_config.get('plugin', [])]
    if plugin_types != ['template_foreground_indicating_mask_generation']:
        raise ValueError('The SPMTrack pipeline needs exactly the plugin '
                         f'template_foreground_indicating_mask_generation, got {plugin_types}')

    visualization = pipeline_config.get('visualization', False)
    online_template_config = pipeline_config.get('online_template', {})
    print('pipeline: SPMTrack one stream tracker (three templates, query state, no CTR)')
    print('visualization: ', visualization)

    main_pipeline = SPMTrackOneStream_Evaluation_MainPipeline(
        device, common_config['template_size'], common_config['search_region_size'],
        build_siamfc_search_region_cropping_parameter_provider_factory(pipeline_config['search_region_cropping']),
        build_post_process(pipeline_config['post_process'], common_config, device),
        common_config['interpolation_mode'], common_config['interpolation_align_corners'],
        common_config['normalization'],
        visualization,
        online_template_config.get('area_factor', 2.0))

    from .template_mask import SPMTrackTemplateFeatForegroundMaskGeneration
    
    return [main_pipeline,
            SPMTrackTemplateFeatForegroundMaskGeneration(common_config['template_size'],
                                                         common_config['template_feat_size'], device)]
