import torch
from torch import nn
from .lovasz_losses import lovasz_softmax

def build_loss(att_mask_2d, att_mask_3d, gt2d, gt3d):

    # Leave xyz out
    bce = nn.BCELoss()

    bce_loss1 = bce(att_mask_3d, gt3d)
    bce_loss2 = bce(att_mask_2d, gt2d)

    return bce_loss1 , bce_loss2