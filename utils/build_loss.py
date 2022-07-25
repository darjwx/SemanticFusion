import torch
from torch import nn
from .lovasz_losses import lovasz_softmax

def build_loss(att_mask, gt, idx):

    # Leave xyz out
    bce = nn.BCELoss()
    bce_loss = bce(att_mask.view(-1)[idx.view(-1)], gt)

    l = bce_loss

    return l
