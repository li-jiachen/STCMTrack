# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified from
#   https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/criterion.py
#   https://github.com/facebookresearch/detr/blob/master/models/segmentation.py
import torch


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor
    ):
    
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss
