# 数据集相关
# data_path='E:/Depth/origin-dataset/KITTI'
data_path='/data/coding/origin/KITTI'
split='./splits/kitti/eigen_zhou'
img_ext='.png'
min_depth=0.01
max_depth=80
epoch_size=200
batch_size=8
epochs=200
# 模型相关
method='sc-depth'
no_ssim=False
no_auto_mask=False
# 实验相关
exp_name='sc-depth-kitti-4-9'
# smooth_weight=1e-3
smooth_weight=0.1
no_min_optimize=False
no_dynamic_mask=False
# 优化器
opt='adamw'  # 优化器
lr_scheduler='onecycle'  # 学习率调度器
lr=1e-3
