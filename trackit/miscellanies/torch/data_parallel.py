import torch


def should_use_data_parallel(device: torch.device) -> bool:
    
    from trackit.miscellanies.torch.distributed import is_dist_initialized
    return (
        not is_dist_initialized() and
        device.type == 'cuda' and
        torch.cuda.is_available() and
        torch.cuda.device_count() > 1
    )
