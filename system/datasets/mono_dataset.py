import random

import numpy as np
import torch
from path import Path
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from system.utils import  read_lines,pil_loader


class MonoDataset(Dataset):
    def __init__(self, cfg, stage='train'):
        """
        训练阶段，返回原始图像，增强图像，内参，逆内参
        测试/验证阶段，原始图像，内参，逆内参，深度
        :param cfg:
        :param stage:
        """
        self.data_path = Path(cfg.data_path)  # 数据集路径
        self.cfg = cfg
        # 数据划分
        if stage == 'train':
            self.filenames = read_lines(Path(cfg.split).joinpath('train_files.txt'))  # 文件名列表
        else:
            self.filenames = read_lines(Path(cfg.split).joinpath('val_files.txt'))  # 文件名列表
        # 图像大小
        self.height = cfg.height
        self.width = cfg.width
        # 插值方法
        self.interp = InterpolationMode.LANCZOS
        # 帧序列
        self.frame_idx = cfg.frame_idx
        self.stage = stage
        self.img_ext = cfg.img_ext
        self.loader = pil_loader
        #---------------------------增强方法
        # 转换为Tensor
        self.to_tensor = transforms.ToTensor()
        # 图像增强
        self.brightness = (0.8, 1.2)
        self.contrast = (0.8, 1.2)
        self.saturation = (0.8, 1.2)
        self.hue = (-0.1, 0.1)
        transforms.ColorJitter.get_params(
            self.brightness, self.contrast, self.saturation, self.hue)
        # 重塑图像
        self.resize = transforms.Resize((self.height, self.width),interpolation=self.interp)
        self.load_depth = self.check_depth()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        inputs = {}
        # 是否对图像进行随机增强
        do_color_aug = self.stage=='train' and random.random() > 0.5
        if do_color_aug:
            color_aug = transforms.ColorJitter( self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)
        do_flip = self.stage=='train' and random.random() > 0.5
        # 解析数据行
        line = self.filenames[idx].split()
        folder = line[0]
        frame_index = int(line[1]) if len(line) == 3 else 0
        side = line[2] if len(line) == 3 else 0

        # 存储所有帧图像
        for i in self.frame_idx:
            if i == 's':
                other_side = {"r": "l", "l": "r"}[side]
                img = self.get_color(folder, frame_index, other_side, do_flip)
            else:
                img = self.get_color(folder, frame_index, side, do_flip)
            img = self.resize(img)
            img = self.to_tensor(img)
            aug_img = color_aug(img)
            inputs[('color',i)] = img
            inputs[('color_aug',i)] = aug_img

        # 获取内参矩阵
        K,K3x3 = self.get_intrinsics(folder, frame_index, side,do_flip,self.height, self.width)
        inv_K = np.linalg.pinv(K)
        inputs['K'] = torch.from_numpy(K)
        inputs['K3x3'] = torch.from_numpy(K3x3)
        inputs['inv_K'] = torch.from_numpy(inv_K)

        # 获取深度真实
        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side,do_flip)
            depth_gt = np.expand_dims(depth_gt, 0)
            depth_gt = self.resize(depth_gt)
            inputs['depth_gt'] = torch.from_numpy(depth_gt.astype(np.float32))

        # 如果是立体图像，则存储外变换矩阵
        if 's' in self.frame_idx:
            stereo_T = np.eye(4,dtype=np.float32)
            baseline_sign = -1 if do_flip else 1  # 控制图像是否翻转
            side_sign = -1 if side == 'l' else 1  # 控制左右相机
            stereo_T[0,3] = side_sign*baseline_sign*0.1  # 变换量为10cm 即两个相机轴心距离10cm
            inputs['stereo_T'] = torch.from_numpy(stereo_T)

        return inputs








    def check_depth(self):
        return self.stage=='val' or self.stage == 'test'

    def get_color(self,folder,frame_index,side,do_flip):
        raise NotImplementedError

    def get_depth(self,folder,frame_index,side,do_flip):
        raise NotImplementedError


    def get_intrinsics(self,folder,frame_index,side,do_flip,height,width):
        raise NotImplementedError


