# 数据集相关
data_path='E:/Depth/origin-dataset/KITTI'
# data_path='/data/coding/origin/KITTI'
split='./splits/kitti/eigen_zhou'
img_ext='.png'
min_depth=0.01
max_depth=80
epoch_size=100
batch_size=4
epochs=1000
# 模型相关
method='sql-depth'
num_layers=50
num_features=512
model_dim=32
patch_size=20
# dim_out=128 # number of bins
# query_nums=128 # number of queries, should be less than h*w/p^2
dim_out=64
query_num=64
disable_automasking = False
# 实验相关
exp_name='sql-depth--kitti'
# smooth_weight=1e-3
smooth_weight=0.1
no_min_optimize=False
# 优化器
opt='adamw'  # 优化器
lr_scheduler='onecycle'  # 学习率调度器
lr=1e-3