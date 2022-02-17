import torch
from torch import nn
from .lovasz_losses import lovasz_softmax

def build_loss(fusion, gt, num_classes, weights=None):

    labels_train = fusion[:, :, 0, 3:num_classes+3]
    labels_gt = gt[:, :, 0, :]
    _, labels_gt = torch.max(labels_gt, dim=2)

    if weights is not None:
        cross_loss = nn.CrossEntropyLoss(weight=weights, ignore_index=0)
    else:
        cross_loss = nn.CrossEntropyLoss(ignore_index=0)

    lovasz = lovasz_softmax(nn.functional.softmax(labels_train, dim=1), labels_gt, ignore=0)

    l = cross_loss(labels_train.view(-1, num_classes).float(), labels_gt.view(-1)) + lovasz

    return l
