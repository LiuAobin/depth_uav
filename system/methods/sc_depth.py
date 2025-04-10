import torch
import torch.nn.functional as F
import wandb
from torchmetrics import MeanMetric

from system.core import compute_metrics
from system.loss import  compute_smoothness_loss, photo_and_geometry_loss
from system.methods.base_method import BaseMethod
from system.models.depth_net import DepthNet
from system.models.pose_net import PoseNet
from system.utils import visualize_depth


class SCDepth(BaseMethod):

    def __init__(self, hparams):
        super(SCDepth,self).__init__(hparams)
        assert self.hparams.frame_ids[0] == 0, "frame_ids must start with 0"
        self.num_input_frames = len(self.hparams.frame_ids)
        self.num_pose_frames = 2 # default=2
        self.use_pose_net = not (self.hparams.folder_type == 'pair' and self.hparams.frame_ids == [0])
        self._build_model()

    def _build_model(self):
        """
        自动获取所有 nn.Module 类型的子模块（比如 encoder、decoder、pose）
        """
        self.depth_net = DepthNet(self.hparams.num_layers)
        self.pose_net = PoseNet()

    def _build_metrics(self):
        self.val_metrics = {
            'de/abs_diff': MeanMetric().to(self.device),
            'de/abs_rel': MeanMetric().to(self.device),
            'de/sq_rel': MeanMetric().to(self.device),
            'de/rmse': MeanMetric().to(self.device),
            'de/log10': MeanMetric().to(self.device),
            'de/rmse_log': MeanMetric().to(self.device),
            'da/a1': MeanMetric().to(self.device),
            'da/a2': MeanMetric().to(self.device),
            'da/a3': MeanMetric().to(self.device),
        }

    def on_train_start(self) -> None:
        self._build_metrics()
        self.log_vis_data = {}

    def forward(self,image):
        return self._predict_depth(image)

    def training_step(self, batch, batch_idx):
        # 准备输出
        batch_out = {}
        # 数据的设备
        for key,value in batch.items():
            batch[key] = value.to(self.device)
        # 预测深度
        for f_i in self.hparams.frame_ids:
            batch_out['depth',f_i] = self._predict_depth(batch['color_aug',f_i])

        self._predict_poses(batch,batch_out)
        losses = self._compute_losses(batch,batch_out)
        self.log_dict({
            f'train/{key}':value for key,value in losses.items()
        } , logger=True, prog_bar=False, on_step=True)
        return losses['loss']

    def validation_step(self, batch, batch_idx):
        for key,value in batch.items():
            batch[key] = value.to(self.device)
        # 深度验证
        if self.hparams.val_mode =='depth':
            disp = self._predict_depth(batch['color',0])
            disp = F.interpolate(disp,
                                 [self.hparams.height, self.hparams.width],
                                 mode='bilinear',
                                 align_corners=False)
            pred_depth = disp
            depth_gt = batch['depth_gt']
            depth_gt = depth_gt.squeeze(1) # 移除通道维度

            metrics = compute_metrics(depth_gt,pred_depth,dataset=self.hparams.dataset_name)
            for key,value in metrics.items():
               self.val_metrics[key].update(value)
            # 可视化
            if batch_idx % 50 == 0:
                self._log_vis_img(batch['color',0], depth_gt, pred_depth, batch_idx)

    def on_validation_epoch_end(self):
        avg_metrics = {key: metric.compute() for key, metric in self.val_metrics.items()}
        # 记录日志
        self.log_dict(avg_metrics)
        self.logger.experiment.log(self.log_vis_data)

        # 记录损失，用于保存模型
        if self.hparams.val_mode == 'depth':
            self.log('val_loss', avg_metrics['de/abs_diff'], on_epoch=True, on_step=False, logger=False)

        # 重置验证指标
        for metric in self.val_metrics.values():
            metric.reset()
        self.log_vis_data = {}

    def on_test_start(self):
        for metrics in self.val_metrics.values():
            metrics.reset()

    def test_step(self, batch, batch_idx):
        for key,value in batch.items():
            batch[key] = value.to(self.device)
        # 深度验证
        if self.hparams.val_mode =='depth':
            disp = self._predict_depth(batch['color',0])
            disp = F.interpolate(disp,
                                 [self.hparams.height, self.hparams.width],
                                 mode='bilinear',
                                 align_corners=False)
            pred_depth = disp
            depth_gt = batch['depth_gt']
            depth_gt = depth_gt.squeeze(1) # 移除通道维度

            metrics = compute_metrics(depth_gt,pred_depth,dataset=self.hparams.dataset_name)
            for key,value in metrics.items():
               self.val_metrics[key].update(value)
            # 可视化
            if batch_idx % 50 == 0:
                self._log_vis_img(batch['color',0], depth_gt, pred_depth, batch_idx)

    def on_test_epoch_end(self):
        avg_metrics = {key: metric.compute() for key, metric in self.val_metrics.items()}
        # 记录日志
        self.log_dict(avg_metrics)
        self.logger.experiment.log(self.log_vis_data)

    def _predict_depth(self,image):
        pred = self.depth_net(image)
        self.log('l',pred.min(),prog_bar=True,logger=False)
        self.log('h', pred.max(), prog_bar=True,logger=False)
        return pred

    def _predict_poses(self,inputs,outputs):
        # 构造输入
        pose_feats = {f_i: inputs['color_aug', f_i] for f_i in self.hparams.frame_ids}

        # 预测每两帧之间的位姿
        for f_i in self.hparams.frame_ids[1:]:
            outputs['pose', 0, f_i] = self.pose_net(pose_feats[0], pose_feats[f_i])
            outputs['pose', f_i, 0] = self.pose_net(pose_feats[f_i], pose_feats[0])

    def _compute_losses(self,inputs,outputs):
        losses = {}
        target = inputs['color', 0]
        depth = outputs['depth', 0]

        photo_loss, geometry_loss = photo_and_geometry_loss(inputs, outputs, hparams=self.hparams)

        smooth_loss = compute_smoothness_loss(depth, target)
        loss = self.hparams.smooth_weight * smooth_loss + 0.1 * geometry_loss + photo_loss
        losses['loss'] = loss
        losses['photo_loss'] = photo_loss
        losses['geometry_loss'] = geometry_loss
        losses['smooth_loss'] = smooth_loss
        return losses

    def _log_vis_img(self, image, gt_depth, pred_depth, batch_idx):
        image = image[0].cpu()
        # 处理 gt_depth 为空的情况
        if gt_depth is not None:
            gt_depth = visualize_depth(gt_depth[0])
            gt_depth_img = wandb.Image(gt_depth, caption="GT Depth")
            self.log_vis_data[f"gt_depth_{batch_idx}"] = gt_depth_img

        # 处理预测深度图
        pred_depth = visualize_depth(pred_depth[0,0])

        # 组织 WandB 记录数据
        self.log_vis_data[f"input_image_{batch_idx}"] = wandb.Image(image, caption="Image")
        self.log_vis_data[f"uav_depth_{batch_idx}"] = wandb.Image(pred_depth, caption="UAV Depth")



