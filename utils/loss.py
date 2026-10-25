import torch
import torch.nn as nn
import torch.nn.functional as F
smoothLoss = nn.SmoothL1Loss(reduction='mean')
bce = nn.BCELoss(reduction='mean')
def iou_loss(predicted, target):
    intersection = (predicted * target).sum()
    union = predicted.sum() + target.sum() - intersection
    iou = intersection / union
    loss = 1 - iou  # 使用 1 减去 IoU 作为损失
    return loss

class BoundaryLoss(nn.Module):
    def __init__(self):
        super(BoundaryLoss, self).__init__()

    def forward(self, prediction, target):
        # 计算梯度信息
        dx = torch.abs(prediction[:, :, :, :-1] - prediction[:, :, :, 0:])
        dy = torch.abs(prediction[:, :, :-1, :] - prediction[:, :, 0:, :])
        grad_pred = dx + dy

        dx = torch.abs(target[:, :, :, :-1] - target[:, :, :, 0:])
        dy = torch.abs(target[:, :, :-1, :] - target[:, :, 0:, :])
        grad_target = dx + dy

        # 计算 Boundary Loss
        boundary_loss = torch.mean(torch.abs(grad_target - grad_pred))

        return boundary_loss

bl = BoundaryLoss()

def multi_bce(preds, gt):
    m_loss = bce(preds[4], gt) + iou_loss(preds[4], gt)
    loss = 0.
    for i in range(0, len(preds) - 1):
        loss += (bce(preds[i], gt) + iou_loss(preds[i], gt)) * ((2 ** i) / 16)# loss
        # loss += bce(preds[i], gt) * ((1+i) / 4)
    return loss + m_loss, m_loss

def MSPNet_multi_bce(preds, gt):
    # print("preds shape:", preds.shape)
    m_loss = bce(preds[1], gt) + iou_loss(preds[1], gt)
    loss = 0.
    for i in range(0, len(preds) - 1):
        loss += bce(preds[i], gt) + iou_loss(preds[i], gt)# loss
        # loss += bce(preds[i], gt) * ((1+i) / 4)
    return loss + m_loss, m_loss

def BLMPM_multi_bce(preds, gt):
    # loss = 0.
    # for i in range(0, len(preds) - 1):
    #     loss += bce(preds[i], gt) + iou_loss(preds[i], gt)
    loss = bce(preds, gt) + iou_loss(preds, gt)
    return loss

def single_bce(pred, gt):
    return bce(pred, gt)


def structure_loss(pred, mask):
    """
    loss function (ref: F3Net-AAAI-2020)
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    # pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def multi_structure_loss(preds, gt):
    m_loss = structure_loss(preds[3], gt)
    loss = 0.
    for i in range(0, len(preds) - 1):
        # loss += bce(preds[i], gt) * ((2 ** i) / 16)  # loss
        loss += (structure_loss(preds[i], gt)) * ((1+i) / 4)
    return loss + m_loss, m_loss