import os
import time
from abc import ABC

import torch
import wandb
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers.wandb import WandbLogger


class BaseSystem(ABC):
    MODEL_STAGE = ["rec", "conv", "pretrain"]

    def __init__(self, opt):
        self.validate_cuda_availability()
        self.csv_log_path = "csv_logs"
        self.opt = opt
        self.num_gpu = torch.cuda.device_count()
        self.gpu_strategy = "ddp" if self.num_gpu > 1 else None

        self.task = opt["task"]
        self.gradient_clip_val = opt.get("gradient_clip", None)
        self.model_name = opt['model']
        self.additional_notes = opt['additional_notes']
        self.sweep = opt['sweep']
        self.logger_type = opt['logger']
        self.dataset_name = opt['dataset']

        self.initialize_task_specific_configs()
        self.setup_logging()

    @staticmethod
    def validate_cuda_availability():
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is not available! You need GPU to train this model!')

    def validate_model_stage(self, model_stage):
        assert model_stage in self.MODEL_STAGE, \
            f"model type is not correct, should be rec, conv or pretrain. {model_stage} is not supported"

    def get_monitor_mode(self, train_stage):
        if train_stage == "conv":
            monitor = f"val/{train_stage}/epoch_loss"
            mode = "min"
        else:
            monitor = f"val/{train_stage}/target_metric" if self.dataset_name == "redial" else f"val/{train_stage}/HitRate@1"
            mode = "max"
            # monitor = f"val/{model_type}/epoch_loss"
            # mode = "min"
        return monitor, mode

    def setup_logging(self):
        self.log_name = f"{self.additional_notes}_{self.model_name}_{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}"
        # this is for generation outputs
        self.gen_output_dir = os.path.join(self.csv_log_path, self.log_name)
        os.makedirs(self.gen_output_dir, exist_ok=True)

        if self.sweep:
            self.logger = WandbLogger()
        elif self.logger_type == "wandb":
            self.logger = WandbLogger(project=self.opt["project_name"],
                                      group=self.opt["group_name"],
                                      tags=self.opt["tags"],
                                      job_type=self.opt["job_type"],
                                      config=self.opt)
            if wandb.run:
                # save code to wandb
                wandb.run.log_code("src/")
        else:
            self.logger = CSVLogger(save_dir=self.csv_log_path, name=self.log_name)

    def initialize_task_specific_configs(self):
        if self.task in ["rec", "crs"]:
            self.init_rec_config(self.opt['rec'])
        if self.task in ["gen", "crs"]:
            self.init_conv_config(self.opt['conv'])

    def init_rec_config(self, rec_opt):
        self.rec_optim_opt = rec_opt
        self.rec_lr = rec_opt['lr']
        self.rec_weight_decay = rec_opt['weight_decay']
        self.rec_epochs = rec_opt['epochs']
        self.num_cycles = rec_opt.get('num_cycles', 1)
        self.rec_batch_size = rec_opt['batch_size']
        self.rec_early_stop_patience = rec_opt['early_stop_patience']
        self.rec_reload = rec_opt["rec_reload"]
        self.rec_model_path = rec_opt["rec_model_path"] if self.rec_reload else None

    def init_conv_config(self, conv_opt):
        self.conv_optim_opt = conv_opt
        self.conv_lr = conv_opt['lr']
        self.conv_weight_decay = conv_opt['weight_decay']
        self.conv_epochs = conv_opt['epochs']
        self.conv_batch_size = conv_opt.get('batch_size', 64)
        self.conv_early_stop_patience = conv_opt.get('early_stop_patience', 5)
        self.conv_reload = conv_opt["conv_reload"]
        self.conv_model_path = conv_opt["conv_model_path"] if self.conv_reload else None

    def build_checkpoint_callback(self, train_stage):
        self.validate_model_stage(train_stage)
        monitor, mode = self.get_monitor_mode(train_stage)

        save_on_train_epoch_end = train_stage not in ["rec", "conv"]
        output_dir = os.path.join(self.csv_log_path, self.log_name, train_stage, "checkpoints")

        return ModelCheckpoint(
            monitor=monitor,
            save_top_k=1,  # save disk space
            save_last=True,
            save_weights_only=True,
            auto_insert_metric_name=True,
            dirpath=output_dir,
            every_n_epochs=1,
            save_on_train_epoch_end=save_on_train_epoch_end,
            mode=mode, )

    def build_early_stop_callback(self, train_stage):
        self.validate_model_stage(train_stage)
        monitor, mode = self.get_monitor_mode(train_stage)
        patience = self.rec_early_stop_patience if train_stage == "rec" else self.conv_early_stop_patience

        return EarlyStopping(
            monitor=monitor,
            patience=patience,
            mode=mode,
            strict=True,
            check_on_train_epoch_end=False,)

    def fit(self):
        """fit the crs models"""
        pass
