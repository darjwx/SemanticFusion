import torch
from torch import nn
from .lovasz_losses import lovasz_softmax

def build_loss(fusion, gt, num_classes, weights=None):

    labels_train = fusion[:, :, 0, 3:num_classes+3]
    labels_gt = gt[:, :, 0, :]

    if weights is not None:
        bce = nn.BCELoss(weight=weights)
    else:
        bce = nn.BCELoss()

    bce_loss = bce(labels_train.view(-1, num_classes), labels_gt.view(-1, num_classes))

    # Onehot -> class indices
    _, labels_gt = torch.max(labels_gt, dim=2)
    lovasz = lovasz_softmax(labels_train, labels_gt, ignore=0)

    l = lovasz + bce_loss

    return l
