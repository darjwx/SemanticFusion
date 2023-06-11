from torch import nn

def build_loss(att_mask, gt, unified_mask):

    if unified_mask:
        # BCE loss
        bce = nn.BCELoss()
        loss = bce(att_mask, gt)
    else:
        # CrossEntropy loss
        cross = nn.CrossEntropyLoss()
        loss = cross(att_mask, gt.long())

    return loss