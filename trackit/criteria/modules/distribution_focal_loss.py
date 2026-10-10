from torch import nn
import torch.nn.functional as F


def distribution_focal_loss(pred, label):
    
    dis_left = label.long()
    dis_right = dis_left + 1
    weight_left = dis_right.float() - label
    weight_right = label - dis_left.float()
    loss = F.cross_entropy(pred, dis_left, reduction='none') * weight_left \
        + F.cross_entropy(pred, dis_right, reduction='none') * weight_right
    return loss


class DistributionFocalLoss(nn.Module):
    

    def __init__(self):
        super(DistributionFocalLoss, self).__init__()

    def forward(self,
                pred,
                target):
        
        loss_cls = self.loss_weight * distribution_focal_loss(
            pred, target)
        return loss_cls
