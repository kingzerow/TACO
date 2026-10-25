import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
# from feature_loss import FeatureLoss
# criterion = torch.nn.CrossEntropyLoss(weight=None, ignore_index=255, reduction='mean').cuda()
# loss_lsc = FeatureLoss().cuda()
# loss_lsc_kernels_desc_defaults = [{"weight": 1, "xy": 6, "rgb": 0.1}]
# loss_lsc_radius = 5
# l = 0.3

def set_gpu(gpu_id='0'):
    if gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')
    elif gpu_id == '1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('USE GPU 1')
    elif gpu_id == '2':
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        print('USE GPU 2')
    elif gpu_id == '3':
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        print('USE GPU 3')


def structure_loss(pred, mask):
    """
    loss function (ref: F3Net-AAAI-2020)
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):]))
              

def load_model_params(model, params_path):
    assert os.path.exists(params_path)
    checkpoints = torch.load(params_path)
    # print(checkpoints['epoch'])

    model.load_state_dict(checkpoints['state_dict'])
    return model


@torch.no_grad()
def distributed_sinkhorn(out):
    Q = torch.exp(out / 10).t()  # Q is K-by-B for consistency with notations from our paper # 0.05 is epsilon
    B = Q.shape[1] * 1  # number of samples to assign # 1 is the samples to assign which use to dispatch assginment
    K = Q.shape[0]  # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    Q /= sum_Q

    for it in range(5):# 3 this is the sinkhorn iteration
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B  # the columns must sum to 1 so that Q is an assignment
    return Q.t()

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class BarlowTwinsLoss(nn.Module):
    def __init__(self, projection_dim, lambd):
        super(BarlowTwinsLoss, self).__init__()
        self.lambd = lambd

        sizes = [2048, 8192, 8192, 8192]
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)
        self.bn = nn.BatchNorm1d(sizes[-1], affine=True)
    def forward(self, z1, z2):
        z1 = self.projector(z1)
        z2 = self.projector(z2)
        B, _ = z1.shape
        c = self.bn(z1).T @ self.bn(z2)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = self.lambd * (on_diag + self.lambd * off_diag)
        return loss

# def BlockLoss(image, out2, out3, out4, out5, out6, fg_label, bg_label, loss_intra):
#     image_ = F.interpolate(image, scale_factor=0.25, mode='bilinear', align_corners=True)
#     ref_sample = {'rgb': image_}
#     # print('sample :', image_.max(), image_.min(), image_.std())
#     ref_out2_ = F.interpolate(out2[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)
#     ref_loss2_lsc = \
#         loss_lsc(ref_out2_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, ref_sample, image_.shape[2],
#                  image_.shape[3])['loss']
#     loss2 = criterion(out2, fg_label) + criterion(out2, bg_label) + l * ref_loss2_lsc + \
#                 loss_intra[0]
#     ###### ref auxiliary losses ######
#     ref_out3_ = F.interpolate(out3[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)
#     ref_loss3_lsc = \
#         loss_lsc(ref_out3_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, ref_sample, image_.shape[2],
#                  image_.shape[3])['loss']
#     loss3 = criterion(out3, fg_label) + criterion(out3, bg_label) + l * ref_loss3_lsc + \
#                 loss_intra[1]
#     ref_out4_ = F.interpolate(out4[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)
#     ref_loss4_lsc = \
#         loss_lsc(ref_out4_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, ref_sample, image_.shape[2],
#                  image_.shape[3])['loss']
#     loss4 = criterion(out4, fg_label) + criterion(out4, bg_label) + l * ref_loss4_lsc + \
#                 loss_intra[2]
#     ref_out5_ = F.interpolate(out5[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)
#     ref_loss5_lsc = \
#         loss_lsc(ref_out5_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, ref_sample, image_.shape[2],
#                  image_.shape[3])['loss']
#     loss5 = criterion(out5, fg_label) + criterion(out5, bg_label) + l * ref_loss5_lsc + \
#                 loss_intra[3]
#     ref_out6_ = F.interpolate(out6[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)
#     ref_loss6_lsc = \
#         loss_lsc(ref_out6_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, ref_sample, image_.shape[2],
#                  image_.shape[3])['loss']
#     loss6 = criterion(out6, fg_label) + criterion(out6, bg_label) + l * ref_loss6_lsc + \
#                 loss_intra[4]
#     return loss2, loss3, loss4, loss5, loss6