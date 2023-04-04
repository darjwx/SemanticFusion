import torch
from torch import nn
from .lovasz_losses import lovasz_softmax

def build_loss(att_mask, gt2d, gt3d):

    # Leave xyz out
    bce = nn.BCELoss()

    bce_loss1 = bce(att_mask[:,0], gt3d)
    bce_loss2 = bce(att_mask[:,1], gt2d)

    return bce_loss1 , bce_loss2