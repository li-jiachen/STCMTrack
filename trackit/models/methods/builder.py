from torch import nn
from trackit.models import ModelBuildingContext


def get_model_build_context(config: dict) -> ModelBuildingContext:
    if config['type'] == 'STCMTrack':
        from .STCMTrack.builder import get_STCMTrack_build_context
        build_context = get_STCMTrack_build_context(config)
    elif config['type'] == 'SPMTrack':
        # Independent baseline (official SPMTrack structure); shares no model code with STCMTrack.
        from .SPMTrack.builder import get_SPMTrack_build_context
        build_context = get_SPMTrack_build_context(config)
    else:
        raise NotImplementedError()
    if isinstance(build_context, nn.Module):
        model = build_context
        build_context = ModelBuildingContext(lambda _: model, lambda _: model.__class__.__name__, None)
    return build_context
