
import os
import numpy as np
from omegaconf import DictConfig
import hydra
from tqdm.auto import tqdm

import wandb
import matplotlib.pyplot as plt

import torch
from timm.utils.model_ema import ModelEmaV3

from dataset import build_transform, build_dataset
from dataset.config import SeriesDataConfig
from evaluator import build_evaluator
from utils.log import logger
from utils.metrics import compute_final_score
from utils.optim import LearningRater


class DetectorTrainer:
    def __init__(self, cfg: DictConfig, data_config: SeriesDataConfig, device: torch.device|str
                 , model: torch.nn.Module, fold_index: int):
        self.save_folder = cfg.paths.save_folder
        self.model_name = cfg.model.backbone.backbone_name
        self.num_fold = cfg.data.num_fold
        self.max_epoch = cfg.trainer.max_epoch
        self.device = device

        self.data_config = data_config
        self.fold_index = fold_index

        transform = build_transform(cfg.data.nifti_transform)
        self.train_dataset = \
            build_dataset(cfg, self.data_config, self.fold_index, transform, is_train=True)
        self.train_dataloader = \
            torch.utils.data.DataLoader(self.train_dataset, cfg.data.batch_size
                                        , shuffle=True, drop_last=True)
        
        self.clip_grad = 35
        self.grad_scaler = torch.GradScaler(enabled=cfg.trainer.fp16)

        self.criterion = torch.nn.BCEWithLogitsLoss().to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1)
        lrfn: LearningRater = hydra.utils.instantiate(cfg.trainer.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, lr_lambda=lrfn)
        # self.scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer=self.optimizer, factor=0.00001
        #                                                      , total_iters=self.max_epoch)

        self.evaluator = build_evaluator(cfg, self.data_config, self.device, self.fold_index)
        
        self.model_ema: ModelEmaV3|None = None
        self.ema_update_start_step: int = 0
        if cfg.trainer.get('ema') and cfg.trainer.ema.get('enable') and cfg.trainer.ema.enable:
            self.model_ema = ModelEmaV3(model, decay=cfg.trainer.ema.decay, device=self.device)
            self.ema_update_start_step = cfg.trainer.ema.update_start_step

        self.start_epoch = 0

    def train(self, model: torch.nn.Module):
        is_autocast_enabled = torch.is_autocast_enabled(str(self.device))
        autocast_dtype = torch.get_autocast_dtype(str(self.device))
        print(f'is_autocast_enabled: {is_autocast_enabled}, autocast_dtype: {autocast_dtype}')
        
        # torch.autograd.set_detect_anomaly(True)

        self.train_losses = []
        self.valid_losses = []

        os.makedirs(self.save_folder, exist_ok=True)

        model_save_folder = os.path.join(self.save_folder, f'{self.model_name}_{self.data_config.save_start_time_str}')
        os.makedirs(model_save_folder, exist_ok=True)

        self.best_checkpoint_name \
            = f'best_checkpoint_{self.model_name}_{self.fold_index:02d}_{self.data_config.save_start_time_str}.pth'
        
        for epoch in range(self.start_epoch, self.max_epoch):
            model.train()

            train_loss, train_accuracy = self.train_one_epoch(model, epoch)

            valid_loss, valid_accuracy = self.eval(model)

            self.train_losses.append(train_loss)
            self.valid_losses.append(valid_loss)
            
            logger.info(f'Fold {self.fold_index+1}/{self.num_fold}, Epoch {epoch+1}/{self.max_epoch}:')
            logger.info(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
            logger.info(f'Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_accuracy:.4f}')

            wandb.log(
                {
                    "train_loss": train_loss,
                    "train_acc": train_accuracy,
                    "valid_loss": valid_loss,
                    "valid_acc": valid_accuracy,
                }
                , step=epoch+1)
            
            if self.evaluator.update_best_valid_loss(valid_loss):
                logger.info(f'Validation loss improved from {self.evaluator.prev_best_valid_loss:.4f} '
                      f'to {self.evaluator.best_valid_loss:.4f}.')
                logger.info('Save checkpoint.')
                torch.save(model.state_dict(), os.path.join(model_save_folder, self.best_checkpoint_name))

    def eval(self, model: torch.nn.Module) -> tuple[float, float]:
        model_eval = model if self.model_ema is None else self.model_ema
        model_eval.eval()
        
        # Evaluate
        valid_loss, valid_accuracy = self.evaluator.evaluate(model_eval, self.criterion)

        return valid_loss, valid_accuracy

    def train_one_epoch(self, model: torch.nn.Module, epoch: int) -> tuple[float, float]:
        train_loss = 0
        train_logits: list[np.ndarray] = []
        train_labels: list[np.ndarray] = []

        epoch_size = len(self.train_dataloader)
        optimizer_step_skip_start_batch_index = int(-1)
        grad_norm_isnan = False
        grad_norm_isinf = False

        for batch_index, batch_data in enumerate(tqdm(self.train_dataloader)):
            forward_step = batch_index + (epoch * epoch_size)

            images, labels = batch_data
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            with torch.autocast(device_type=str(self.device)):
                pred = model(images)
                cost = self.criterion(pred, labels)
            
            self.grad_scaler.scale(cost).backward()

            grad_norm = None
            if self.clip_grad > 0:
                # unscale gradients
                self.grad_scaler.unscale_(self.optimizer)
                # clip gradients
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.clip_grad)
                if torch.logical_or(grad_norm.isnan(), grad_norm.isinf()) \
                    and 0 > optimizer_step_skip_start_batch_index:
                    optimizer_step_skip_start_batch_index = batch_index
                    grad_norm_isnan = grad_norm.isnan() or grad_norm_isnan
                    grad_norm_isinf = grad_norm.isinf() or grad_norm_isinf
            # optimizer.step
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.optimizer.zero_grad()

            if self.model_ema is not None and forward_step >= self.ema_update_start_step:
                self.model_ema.update(model)

            # min_scale = 128
            # if self.grad_scaler._scale < min_scale:
            #     self.grad_scaler._scale = torch.tensor(min_scale).to(self.grad_scaler._scale)
            
            train_loss += cost.item()
            train_labels.append(labels.detach().cpu().numpy())
            train_logits.append(pred.sigmoid().detach().cpu().numpy())

            # if batch_index > 4:
            #     break
        
        train_accuracy: float = 0
        if 0 < len(train_labels) and 0 < len(train_logits):
            train_y_true = np.concatenate(train_labels, axis=0)
            train_y_score = np.concatenate(train_logits, axis=0)
            train_accuracy = compute_final_score(train_y_true, train_y_score)
        train_loss /= len(self.train_dataloader)

        self.scheduler.step()
        
        if grad_norm_isnan or grad_norm_isinf:
            # scaler is going to skip optimizer.step() if grads are nan or inf
            # some updates are skipped anyway in the amp mode, but we can count for statistics
            print(f'Skiped optimizer step, start_batch_index: {optimizer_step_skip_start_batch_index}'
                  f', isnan: {grad_norm_isnan}, isinf: {grad_norm_isinf}')

        return train_loss, train_accuracy
    
    def show_loss_and_epoch_graph(self):
        plt.plot(self.train_losses, color="blue", label='train loss')
        plt.plot(self.valid_losses, color="red", label='valid loss')
        plt.legend(loc='upper right')
        plt.xticks(np.arange(0, len(self.train_losses), 2))
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.grid()
        plt.show()


def build_trainer(cfg: DictConfig, data_config: SeriesDataConfig, device: torch.device|str
                  , model: torch.nn.Module, fold_index: int=0) \
    -> DetectorTrainer:
    return DetectorTrainer(cfg, data_config, device, model, fold_index)
