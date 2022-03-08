import torch
from torch import nn
from .lovasz_losses import lovasz_softmax

def build_loss(fusion, gt, num_classes, idx, weights=None):

    # Leave xyz out
    train = fusion[:, :, :, 3:num_classes+3]

    if weights is not None:
        bce = nn.BCELoss(weight=weights)
    else:
        bce = nn.BCELoss()

    #bce_loss = bce(train.view(-1, num_classes), gt.view(-1, num_classes))

    # Onehot -> class indices
    _, labels_gt = torch.max(gt, dim=3)

    lovasz = lovasz_softmax(train, labels_gt, idx=idx, ignore=0)

    l = lovasz # + bce_loss

    return l
