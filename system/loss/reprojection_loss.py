import torch
from .ssim import SSIM

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
ssim = SSIM().to(device)


def compute_reprojection_loss(pred,target,no_ssim=False):
    """
    计算预测图像和目标图像之间的重投影损失
    Args:
        pred: 预测图像(重投影的结果) of shape [B,C,H,W]
        target: 目标图像(源图像) of shape [B,C,H,W]
        no_ssim:
    Returns:
        torch.Tensor: 计算得到的重投影损失
    """
    # L1损失
    abs_diff = torch.abs(pred - target)
    l1_loss = abs_diff.mean(1,True)  # 计算通道维度的均值，保持维度
    if no_ssim:
        return l1_loss
    else:
        ssim_loss = ssim(pred,target)
        ssim_loss = ssim_loss.mean(1,True)
        reprojection_loss =  0.85 * ssim_loss + 0.15 * l1_loss
        return reprojection_loss