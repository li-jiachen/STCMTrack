from torch.optim.swa_utils import AveragedModel


# https://github.com/pytorch/vision/blob/main/references/classification/utils.py/#L159-170
class ExponentialMovingAverage(AveragedModel):
    

    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)
