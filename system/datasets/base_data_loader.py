from pytorch_lightning import LightningDataModule
from torch.utils.data import RandomSampler, DataLoader

from system.datasets.kitti_dataset import KittiDataset

dataset_map = {
    'kitti': KittiDataset,
}

class BaseDataModule(LightningDataModule):
    def __init__(self, cfg):
        """
        初始化模块
        :param cfg: 配置信息
                - cfg.dataset_name: 数据集名称。
                - cfg.dataset_dir: 数据集的根目录路径。
                - cfg.val_mode: 验证模式，'depth'或'photo'。
                - cfg.batch_size: 批次大小。
                - cfg.folder_type: 数据类型。sequence or pair。
        """
        super().__init__()
        print(f'datasets---->init dataset transform...')
        # 保存超参数
        self.save_hyperparameters()
        self.cfg = cfg

    def setup(self, stage: str) -> None:
        """
       设置数据集和数据加载器，根据训练或者验证模式选择适当的数据集
       :param stage:(str, optional) 数据准备阶段。可用于多阶段处理（如训练、验证或测试阶段）
       :return:
       """
        print(f'datasets---->setup dataset of {self.cfg.dataset_name}...')
        # 训练数据集
        self.train_dataset = self.get_train_dataset()
        self.val_dataset = self.get_val_dataset()
        # 加载测试数据集
        self.test_dataset = self.get_test_dataset()

    def get_train_dataset(self):
        if self.cfg.dataset_name in dataset_map.keys():
            return dataset_map[self.cfg.dataset_name](self.cfg,'train')

    def get_val_dataset(self):
        if self.cfg.dataset_name in dataset_map.keys():
            return dataset_map[self.cfg.dataset_name](self.cfg,'val')

    def get_test_dataset(self):
        if self.cfg.dataset_name in dataset_map.keys():
            return dataset_map[self.cfg.dataset_name](self.cfg,'test')

    def train_dataloader(self):
        """
        训练数据加载器
        :return:
        :rtype:
        """
        sampler = RandomSampler(self.train_dataset,
                                replacement=True,  # 运行替换采样
                                num_samples=self.cfg.batch_size * self.cfg.epoch_size)  # 计算需要的样本数量
        return DataLoader(self.train_dataset,  # 数据集
                          batch_size=self.cfg.batch_size,  # 批次大小
                          num_workers=0,  # 加载数据时使用的线程数
                          pin_memory=True,  # 使用固定内存，提升数据加载速度
                          sampler=sampler,  # 随机采样
                          drop_last=True,  # 丢弃最后一个不完整的批次
                          )


    def val_dataloader(self):
        """
        验证数据加载器
        :return:
        :rtype:
        """
        return DataLoader(self.val_dataset,  # 使用验证数据集
                          shuffle=False,  # 打乱数据
                          num_workers=0,  # 加载数据的工作线程数
                          # num_workers=0,
                          batch_size=self.cfg.batch_size,  # 设置批次大小
                          pin_memory=True)  # 使用固定内存，提升数据加载速度

    def test_dataloader(self):
        """
        测试数据加载器
        :return:
        :rtype:
        """
        return DataLoader(self.test_dataset,  # 使用验证数据集
                          shuffle=False,  # 不打乱数据
                          num_workers=0,  # 加载数据的工作线程数
                          batch_size=self.cfg.batch_size,  # 设置批次大小
                          pin_memory=True)  # 使用固定内存，提升数据加载速度
