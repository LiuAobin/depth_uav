# 深度估计相关实验

1. 已完成：配置文件加载和读取
2. KITTI和cityscapes数据集的预处理

##  [数据集](./data/DATA.md)

## Model
TAU:Squeeze-and-excitation networks
VAN:Visual Attention Network
![img.png](imgs/img.png)
![img.png](imgs/img2.png)

![img.png](imgs/img3.png)
# 训练
## MidAir数据集
```shell
python -m tools.train --config_file ./configs/sc_depth/midair.py
```
# 可视化
Tensorboard
```shell
tensorboard --logdir=./work_dirs --host=0.0.0.0
```
Wandb

api_key:4186dee52f004546f3d3caaa4113d1a907afa21d
```shell
wandb.login(key=[your_api_key])
```

# 要求
## config.dataset
```
--split 是一个文件夹，内两个文件，分别是train_files.txt和val_files.txt,用于划分训练集和测试集
--exp_name 字符串 实验名，用于日志记录
--img_ext 图像文件扩展名(.png)或者其他
--depth_ext 深度图像文件扩展名，如果有则按照这个获取深度
--val_mode 验证模式，是使用深度验证(depth)，还是使用图像验证(photo)(和训练阶段一致)
--folder_type 文件类型 是图像对(pair)的形式或者单目视频(sequence)
--frame_idxs 需要的帧索引，0表示目标帧，其余数据以目标帧为起始点移动
```
