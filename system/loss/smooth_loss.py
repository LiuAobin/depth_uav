import torch


def compute_smoothness_loss(depth, image):
    """
    计算深度图的平滑损失，保持深度的局部平滑性
    使用输入的彩色图像 img 进行边缘感知，使得深度图在图像边缘变化较大的地方可以允许变化较大，
    而在光滑区域保持平滑。
    Args:
        depth:(batch, 1, height, width)
        image:(batch, channels, height, width)
    Returns:
    """
    # 计算深度图梯度(水平和垂直方向)
    mean_depth = depth.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
    norm_depth = depth/(mean_depth+1e-7)
    depth = norm_depth
    # 进行 Min-Max 归一化，保持深度图的原始范围
    # depth_min = depth.min()
    # depth_max = depth.max()
    # norm_depth = (depth - depth_min) / (depth_max - depth_min + 1e-7)
    # depth = norm_depth
    grad_depth_x = torch.abs(depth[:, :, :, :-1] - depth[:, :, :, 1:]) # 水平方向
    grad_depth_y = torch.abs(depth[:, :, :-1, :] - depth[:, :, 1:, :])

    # 计算图像梯度的均值(通道均值)
    grad_img_x = torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])
    grad_img_x = torch.mean(grad_img_x,1,keepdim=True)

    grad_img_y = torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :])
    grad_img_y = torch.mean(grad_img_y,1,keepdim=True)

    # 通过对图像梯度的指数衰减来加权图像梯度，抑制边缘区域的平滑约束
    grad_depth_x *= torch.exp(-grad_img_x)
    grad_depth_y *= torch.exp(-grad_img_y)
    # 返回水平方向和垂直方向的平均损失
    return grad_depth_x.mean()+grad_depth_y.mean()