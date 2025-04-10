import torch
import torch.nn.functional as F
import wandb
from torchmetrics import MeanMetric

from system.core import compute_metrics
from system.loss import compute_reprojection_loss, compute_smoothness_loss, photo_and_geometry_loss
from system.methods.base_method import BaseMethod
from system.models import ResNetEncoderDecoder, DepthDecoderQueryTr, PoseCNN
from system.models.depth_net import DepthNet
from system.models.pose_net import PoseNet
from system.modules import ResNetEncoder
from system.utils import BackProjectDepth, Project3D, transformation_from_parameters, visualize_depth, disp_to_depth, \
    inverse_warp_img


class SQLDepth(BaseMethod):

    def __init__(self, hparams):
        super(SQLDepth,self).__init__(hparams)
        assert self.hparams.frame_ids[0] == 0, "frame_ids must start with 0"
        self.num_input_frames = len(self.hparams.frame_ids)
        self.num_pose_frames = 2 # default=2
        self.use_pose_net = not (self.hparams.folder_type == 'pair' and self.hparams.frame_ids == [0])
        self._build_model()

    def _build_model(self):
        """
        自动获取所有 nn.Module 类型的子模块（比如 encoder、decoder、pose）
        """
        # depth-encoder
        self.depth_encoder = ResNetEncoderDecoder(
            num_layers=self.hparams.num_layers,
            num_features=self.hparams.num_features,
            model_dim=self.hparams.model_dim
        )
        # depth-decoder
        self.depth_decoder = DepthDecoderQueryTr(
            in_channels=self.hparams.model_dim,
            patch_size=self.hparams.patch_size,
            dim_out=self.hparams.dim_out,
            embedding_dim=self.hparams.model_dim,
            query_nums=self.hparams.query_nums,
            num_heads=4,
            min_val=self.hparams.min_depth,
            max_val=self.hparams.max_depth
        )
        # self.depth_net = DispResNet(num_layers=50, pretrained=True)
        # pose-net
        self.pose_net = PoseCNN(2)

    def _build_projection(self):
        """
        构建投影和反投影
        Returns:
        """
        h = self.hparams.height
        w = self.hparams.width
        # 反向投影深度
        self.back_project_depth = BackProjectDepth(
            batch_size=self.hparams.batch_size,
            height=h, width=w).to(self.device)
        # 三维投影
        self.project_3d = Project3D(
            batch_size=self.hparams.batch_size,
            height=h, width=w).to(self.device)

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
        self._build_projection()
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
            batch_out['disp',f_i] = self._predict_depth(batch['color_aug',f_i])

        self._predict_poses(batch,batch_out)
        self._generate_images_pred(batch,batch_out)
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
        features = self.depth_encoder(image)
        pred = self.depth_decoder(features)
        self.log('l',pred.min(),prog_bar=True,logger=False)
        self.log('h', pred.max(), prog_bar=True,logger=False)
        return pred

    def _predict_poses(self,inputs,outputs):
        # outputs = {}
        # 构造输入
        pose_feats = {f_i:inputs['color_aug',f_i] for f_i in self.hparams.frame_ids}

        # 预测每两帧之间的位姿
        for f_i in self.hparams.frame_ids[1:]:
            pose_input = [pose_feats[0],pose_feats[f_i]]
            pose_input = torch.cat(pose_input, dim=1)
            pose = self.pose_net(pose_input)
            outputs['pose',0,f_i] =  pose
            axisangle = pose[..., :3]
            translation = pose[..., 3:]
            outputs[('axisangle', 0, f_i)] = axisangle
            outputs[('translation', 0, f_i)] = translation
            outputs[('cam_T_cam', 0, f_i)] = transformation_from_parameters(
                axisangle[:, 0], translation[:, 0])

    def _generate_images_pred(self,inputs,outputs):
        # 取出深度
        disp = outputs['disp',0]
        disp = F.interpolate(disp,
                             [self.hparams.height,self.hparams.width],
                             mode='bilinear', align_corners=False)
        depth = disp
        outputs['depth',0] = depth

        # 遍历生成投影图像
        for frame_id in self.hparams.frame_ids[1:]:
            if frame_id == 's': # 立体图像对
                T = inputs['stereo_T']
            else:
                T = outputs['cam_T_cam',0,frame_id]

            if self.hparams.folder_type != 'pair':
                axisangle = outputs['axisangle',0,frame_id]
                translation = outputs['translation',0,frame_id]
                inv_depth = 1 / depth
                mean_inv_depth = inv_depth.mean(3,True).mean(2,True)
                T = transformation_from_parameters(
                    axisangle[:,0],
                    translation[:,0]*mean_inv_depth[:,0])
            outputs[('color',frame_id)] = inverse_warp_img(inputs['color',frame_id],
                                                           outputs['depth',0],
                                                           outputs['pose',0,frame_id],
                                                           inputs['K3x3'])
            # cam_points = self.back_project_depth(depth,inputs['inv_K'])
            # pix_coords = self.project_3d(cam_points,inputs['K'],T)
            # outputs[('sample',frame_id)] = pix_coords
            # outputs[('color',frame_id)] = F.grid_sample(
            #     inputs['color',frame_id],
            #     outputs['sample',frame_id],
            #     padding_mode='border',
            #     align_corners=True)
            if not self.hparams.disable_automasking:
                outputs[('color_identity',frame_id)] = inputs['color',frame_id]

    def _compute_losses(self,inputs,outputs):
        losses = {}
        target = inputs['color',0]
        depth = outputs['depth',0]

        reprojection_losses = []
        for frame_id in self.hparams.frame_ids[1:]:
            pred = outputs['color', frame_id]
            reprojection_losses.append(
                compute_reprojection_loss(pred, target))
        reprojection_losses = torch.cat(reprojection_losses, dim=1)

        if not self.hparams.disable_automasking:
            identity_reprojection_losses = []
            for frame_id in self.hparams.frame_ids[1:]:
                pred = inputs['color', frame_id]
                identity_reprojection_losses.append(
                    compute_reprojection_loss(pred, target))
            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)
            identity_reprojection_losses += torch.randn(
                identity_reprojection_losses.shape).to(
                identity_reprojection_losses.device) * 0.00001
            combined = torch.cat((identity_reprojection_losses, identity_reprojection_losses), dim=1)
        else:
            combined = reprojection_losses

        if combined.shape[1] > 1:
            to_optimise = combined
        else:
            to_optimise, idxs = torch.min(combined, dim=1)

        reprojection_loss = to_optimise.mean()

        smooth_loss = compute_smoothness_loss(depth, target)
        loss = self.hparams.smooth_weight*smooth_loss + reprojection_loss
        losses['loss'] = loss
        losses['reprojection_loss'] = reprojection_loss
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



