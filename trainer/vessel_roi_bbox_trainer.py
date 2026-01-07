
import time, os
import numpy as np
from omegaconf import DictConfig
import pydoc
from tqdm.auto import tqdm

import wandb
import torch
from timm.utils.model_ema import ModelEmaV3

from utils.log import logger
from dataset import VesselSegmentDataManager
from evaluator import VesselROIBBoxEvaluator


class AccuracyCalculator:
    def __init__(self) -> None:
        self.reset()
    
    def reset(self):
        self.logits: list[np.ndarray] = []
        self.labels: list[np.ndarray] = []
    
    def enqueue(self, labels: torch.Tensor, logits: torch.Tensor):
        self.labels.append(labels.detach().cpu().numpy())
        self.logits.append(logits.sigmoid().detach().cpu().numpy())

    def calculate(self) -> float:
        accuracy: float = 0
        if 0 < len(self.labels) and 0 < len(self.logits):
            total_labels = np.concatenate(self.labels, axis=0)
            total_logits = np.concatenate(self.logits, axis=0)
            accuracy = np.abs(total_logits - total_labels).mean()
        
        return accuracy


class VesselROIBBoxTrainer:
    def __init__(self
                #  , cfg: DictConfig
                 , fold_index: int
                 , fp16: bool
                 , grad_norm_clip_max: int
                 , max_epoch: int
                 , save_path: str
                 , optimizer: DictConfig
                 , lr_scheduler: DictConfig
                 , criterion: DictConfig
                 , ema: DictConfig
                 , device: torch.device|str
                 , data_manager: VesselSegmentDataManager
                 , model: torch.nn.Module
                 ):
        self.fold_index = fold_index
        self.max_epoch = max_epoch
        self.save_path = save_path

        self.device = device
        self.grad_scaler = torch.GradScaler(enabled=fp16)
        self.grad_norm_clip_max = grad_norm_clip_max

        self.data_manager = data_manager
        self.train_dataloader = self.data_manager.get_train_dataloader()

        self.model = model.to(self.device)

        optim_type = pydoc.locate(optimizer.target_type)
        self.optimizer = optim_type(self.model.parameters(), **optimizer.kwargs)

        lr_scheduler_type = pydoc.locate(lr_scheduler.target_type)
        self.scheduler = lr_scheduler_type(optimizer=self.optimizer, **lr_scheduler.kwargs)

        self.criterion = criterion.mae.to(self.device)

        self.evaluator \
            = VesselROIBBoxEvaluator(self.device, self.data_manager.get_valid_dataloader(), AccuracyCalculator())
        
        self.model_ema: ModelEmaV3|None = None
        self.ema_update_start_step: int = 0
        if ema.enable:
            self.model_ema = ModelEmaV3(self.model, decay=ema.decay, device=self.device)
            self.ema_update_start_step = ema.update_start_step

        self.acc_calculator = AccuracyCalculator()
        self.train_start_time_str = time.strftime('%Y%m%d_%H%M%S')
        self.start_epoch = 0

    def train(self):
        is_autocast_enabled = torch.is_autocast_enabled(str(self.device))
        autocast_dtype = torch.get_autocast_dtype(str(self.device))
        print(f'is_autocast_enabled: {is_autocast_enabled}, autocast_dtype: {autocast_dtype}')
        
        # torch.autograd.set_detect_anomaly(True)

        self.train_losses = []
        self.valid_losses = []

        os.makedirs(self.save_path, exist_ok=True)

        model_save_folder \
            = os.path.join(self.save_path, f'{type(self.model).__name__.lower()}_{self.train_start_time_str}')
        os.makedirs(model_save_folder, exist_ok=True)

        self.best_checkpoint_name \
            = f'best_checkpoint_{type(self.model).__name__.lower()}_{self.fold_index:02d}_{self.train_start_time_str}.pth'
        
        for epoch in range(self.start_epoch, self.max_epoch):
            self.model.train()

            train_loss, train_accuracy = self.train_one_epoch(self.model, epoch)

            valid_loss, valid_accuracy = self.eval(self.model)

            self.train_losses.append(train_loss)
            self.valid_losses.append(valid_loss)
            
            logger.info(f'Fold {self.fold_index+1}/{self.data_manager.num_fold}, Epoch {epoch+1}/{self.max_epoch}:')
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
                torch.save(self.model.state_dict(), os.path.join(model_save_folder, self.best_checkpoint_name))

    def eval(self, model: torch.nn.Module) -> tuple[float, float]:
        model_eval = model if self.model_ema is None else self.model_ema
        model_eval.eval()
        
        # Evaluate
        valid_loss, valid_accuracy = self.evaluator.evaluate(model_eval, self.criterion)

        return valid_loss, valid_accuracy

    def train_one_epoch(self, model: torch.nn.Module, epoch: int) -> tuple[float, float]:
        train_loss = 0
        self.acc_calculator.reset()

        epoch_size = len(self.train_dataloader)
        optimizer_step_skip_start_batch_index = int(-1)
        grad_norm_isnan = False
        grad_norm_isinf = False

        for batch_index, batch_data in enumerate(tqdm(self.train_dataloader)):
            forward_step = batch_index + (epoch * epoch_size)

            volumes, labels, _ = batch_data
            volumes = volumes.to(self.device)
            labels = labels.to(self.device)
            
            with torch.autocast(device_type=str(self.device)):
                pred = model(volumes)
                loss = self.criterion(pred, labels)
            
            self.grad_scaler.scale(loss).backward()

            grad_norm = None
            if self.grad_norm_clip_max > 0:
                # unscale gradients
                self.grad_scaler.unscale_(self.optimizer)
                # clip gradients
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.grad_norm_clip_max)
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
            
            train_loss += loss.item()

            self.acc_calculator.enqueue(labels, pred)

            # if batch_index > 2:
            #     break
        
        train_loss /= len(self.train_dataloader)
        train_accuracy = self.acc_calculator.calculate()

        self.scheduler.step()
        
        if grad_norm_isnan or grad_norm_isinf:
            # scaler is going to skip optimizer.step() if grads are nan or inf
            # some updates are skipped anyway in the amp mode, but we can count for statistics
            print(f'Skiped optimizer step, start_batch_index: {optimizer_step_skip_start_batch_index}'
                  f', isnan: {grad_norm_isnan}, isinf: {grad_norm_isinf}')

        return train_loss, train_accuracy
