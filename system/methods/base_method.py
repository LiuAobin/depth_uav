from pytorch_lightning import LightningModule

from system.core import get_optim_scheduler,timm_schedulers


class BaseMethod(LightningModule):
    def __init__(self, config):
        super().__init__()
        # 保存超参数
        self.save_hyperparameters(config)
        self._build_model()

    def configure_optimizers(self,**models_dict):
        """
        配置优化器和学习率调度器
        Returns:
        """
        models = list(models_dict.values())  # 将所有 model 取出，转为 list
        optimizer, scheduler, by_epoch = get_optim_scheduler(  # 根据超参数设置优化器和学习率调度器
            self.hparams,
            self.hparams.epochs,
            models,
            self.hparams.epoch_size
        )
        return {
            "optimizer": optimizer,
            'lr_scheduler':{
                "scheduler": scheduler,
                "interval": "epoch" if by_epoch else "step"
            }
        }

    def lr_scheduler_step(self, scheduler, metric):
        """
        学习率调度器调整步骤
        Args:
            scheduler (): 调度器
            metric ():
        Returns:
        """
        # 如果是timm提供的调度器，则按epoch调整
        if any(isinstance(scheduler,sch) for sch in timm_schedulers):
            scheduler.step(epoch=self.current_epoch)
        else:  # 根据指标或者默认规则调整
            if metric is None:
                scheduler.step()
            else:
                scheduler.step(metric)

    def _build_model(self):
        """
        构建模型，具体逻辑由子类实现
        Returns:
        """
        raise NotImplementedError