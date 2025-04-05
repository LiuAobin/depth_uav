import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T


def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    """
    可视化深度图，转换为彩色图并返回
    Args:
        depth (torch.Tensor): 输入的深度图，形状为(H,W)
        cmap (cv2.COLORMAP): 可选的openCV颜色映射，默认为JET
    Returns:
        转换后的深度图，尺寸为(3,H,W),范围[0,1]

    """
    x = depth.cpu().numpy()  # 将深度图放到cpu并转换为np数组
    x = np.nan_to_num(x)  # 替换nan为0，避免后续计算出现无效值
    # 获取深度图中的最大值和最小值
    mi = np.min(x)
    ma = np.max(x)
    # 对深度图进行归一化
    x = (x-mi)/(ma-mi+1e-8)  # x_normalized = (x - min) / (max-min)
    # 将深度值从[0,1]范围映射到[0,255]并转换为uint8 便于颜色映射
    x = (255*x).astype(np.uint8)
    # 使用opencv的applyColorMap将灰度深度图转换为伪彩色图
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_)  # 转换为张量，并将图像范围从[0,255]缩放到[0,1]
    return x_