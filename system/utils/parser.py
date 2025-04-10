import argparse
from .config_utils import Config
def create_parser():
    parser = argparse.ArgumentParser(description="深度估计实验相关参数")
    # 基础配置  --config_file是关键
    parser.add_argument('config_file', nargs='?', type=str,
                        default='./configs/darknet/midair.py',
                        help='额外的配置文件所在路径')
    parser.add_argument('--work_dir', type=str,
                        default='./work_dirs',
                        help='工作目录')
    parser.add_argument('--exp_name', type=str,
                        default='kitti_UAV_Depth',
                        help='实验名称，训练或测试过程中所有输出均在{work_dir}/{exp_name}下')
    parser.add_argument('--metric_for_bestckpt', default='val_loss', type=str,
                        help='检查那个损失指标作为保持最佳检查点的信息')
    parser.add_argument('--log_step', type=int,default=1,
                        help='每log_step个Epoch保存一次')
    parser.add_argument('--opt', type=str,
                        default='adamw',
                        help='优化器')
    parser.add_argument('--lr_scheduler', type=str,
                        default='onecycle',
                        help='学习率调度器')
    parser.add_argument('--lr', type=float,
                        default=1e-3,
                        help='初始化学习率')
    parser.add_argument('--weight_decay', type=float,
                        default=1e-4,
                        help='梯度衰减')
    parser.add_argument('--filter_bias_and_bn', type=bool, default=False,
                        help='Whether to set the weight decay of bias and bn to 0')
    # 模型相关
    #-------------------------------- 模型方法 --------------------------------#
    parser.add_argument('--method', type=str,
                        default='sql-depth',
                        help='训练所使用的模型')
    parser.add_argument('--num_layers', type=int,
                        default=50,
                        help='resnet层数')
    parser.add_argument('--num_features', type=int,
                        default=512,
                        help='resnet编码器最终输出层数')
    parser.add_argument('--model_dim',type=int,
                        default=32,
                        help='深度编码器最终输出特征通道数')
    # -------------------------------- 数据集相关 --------------------------------#
    parser.add_argument('--dataset_name', type=str,default='kitti',
                        help='数据集名称')
    parser.add_argument('--batch_size', type=int,
                        default=4,
                        help='每个小批次的大小')
    parser.add_argument('--data_path', type=str,default=None,
                        help='数据集路径')
    parser.add_argument('--split', type=str,default=None,
                        help='数据集分割配置文件位置')
    parser.add_argument('--img_ext', type=str,default='.png',
                        help='图像文件扩展名')
    parser.add_argument('--val_mode', type=str,default='depth',
                        help='验证模式，是使用深度验证(depth)，还是使用图像验证(photo)(和训练阶段一致)')
    parser.add_argument('--folder_type',type=str,default='sequence',
                        help='文件类型 是图像对(pair)的形式或者单目视频(sequence)')
    parser.add_argument('--frame_ids',default=[0,-1,1],
                        help='需要的帧索引，0表示目标帧，其余数据以目标帧为起始点移动')
    parser.add_argument("--height", type=int,
                        default=320,
                        help='图像高度')
    parser.add_argument("--width", type=int,
                        default=1024,
                        help='图像宽度')
    parser.add_argument("--min_depth",
                        type=float,
                        help="minimum depth",
                        default=0.01)
    parser.add_argument("--max_depth",
                        type=float,
                        help="maximum depth",
                        default=80.0)
    parser.add_argument('--channels',type=int,default=3)


    # -------------------------------- 训练相关 --------------------------------#
    parser.add_argument('--epochs', type=int,
                        default=100,
                        help='训练总轮数')
    parser.add_argument('--epoch_size', type=int,
                        default=2000,
                        help='每轮的训练样本数量')
    parser.add_argument('--limit_val_batches', type=float, default=1.0,
                        help='限制验证集的百分比')

    #-------------------------------- 实验配置相关 --------------------------------#
    parser.add_argument('--ckpt_path', default=None, type=str,
                        help="恢复训练的检查点文件")
    parser.add_argument('--resume',
                        action='store_true',
                        help='是否恢复训练')
    parser.add_argument("--num_threads", type=int,
                        default=6,
                        help='处理/加载数据时，使用的线程数')
    parser.add_argument('--test', action='store_true',
                        default=False,
                        help='测试模型')
    parser.add_argument('--no_display_method_info', action='store_true',
                        default=False,
                        help='是否显示方法信息,默认显示')
    parser.add_argument('--fps', default=True, type=bool,
                        help='是否显示推理速度')
    parser.add_argument('--seed', type=int,
                        default=3602,
                        help='随机数种子，确保结果可复现')
    parser.add_argument('--device', type=str,
                        default='cuda',
                        help='使用cpu or cuda 进行张量计算')
    return parser.parse_args()


def load_config(filename: str = None):
    """load and print config"""
    print('loading config from ' + filename + ' ...')
    try:
        configfile = Config(filename=filename)
        config = configfile._cfg_dict
    except (FileNotFoundError, IOError):
        config = dict()
        print('warning: fail to load the config!')
    return config


def update_config(args):
    config = load_config(args.config_file)
    assert isinstance(args, argparse.Namespace) and isinstance(config, dict)
    for k in config.keys():
        if hasattr(args, k):
            print(f'overwrite config key -- {k}: {getattr(args, k)} -> {config[k]}')
            setattr(args, k, config[k])
        else:
            setattr(args, k, config[k])
    return args