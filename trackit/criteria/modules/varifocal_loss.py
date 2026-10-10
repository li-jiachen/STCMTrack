# modified from https://github.com/hyz-xmaster/VarifocalNet/blob/master/mmdet/models/losses/varifocal_loss.py

import torch.nn as nn
import torch.nn.functional as F


def varifocal_loss(pred,
                   target,
                   alpha=0.75,
                   gamma=2.0,
                   iou_weighted=True):
    
    
    assert pred.size() == target.size()
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    if iou_weighted:
        focal_weight = target * (target > 0.0).float() + \
            alpha * (pred_sigmoid - target).abs().pow(gamma) * \
            (target <= 0.0).float()
    else:
        focal_weight = (target > 0.0).float() + \
            alpha * (pred_sigmoid - target).abs().pow(gamma) * \
            (target <= 0.0).float()
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    return loss


class VarifocalLoss(nn.Module):
    def __init__(self,
                 alpha=0.75,
                 gamma=2.0,
                 iou_weighted=True):
        
        super(VarifocalLoss, self).__init__()
        assert alpha >= 0.0
        self.alpha = alpha
        self.gamma = gamma
        self.iou_weighted = iou_weighted

    def forward(self,
                pred,
                target):
        
        return varifocal_loss(
            pred,
            target,
            alpha=self.alpha,
            gamma=self.gamma,
            iou_weighted=self.iou_weighted)
