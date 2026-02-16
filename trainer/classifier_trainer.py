
import time, os, math
from collections import defaultdict
import numpy as np
from omegaconf import DictConfig
import pydoc
from tqdm.auto import tqdm

import wandb
import torch
from timm.utils.model_ema import ModelEmaV3

from utils.log import logger
from utils.metrics import compute_final_score
from dataset import SeriesDataManager
from evaluator import DicomSeriesEvaluator


class ClassifierAccuracyCalculator:
    def __init__(self
                 , predict_save_path: str
                 , current_fold: int
                 ) -> None:
        self.predict_save_path = predict_save_path
        self.current_fold = current_fold

        self.train_start_time_str = time.strftime('%Y%m%d_%H%M%S')

        self.reset()
    
    def reset(self):
        self.logits: list[np.ndarray] = []
        self.labels: list[np.ndarray] = []
        self.uids: list[list[str]] = []
    
    def enqueue(self, labels: torch.Tensor, logits: torch.Tensor, uids: list[str]):
        self.labels.append(labels.detach().cpu().numpy())
        self.logits.append(logits.sigmoid().detach().cpu().numpy())
        self.uids.append(uids)

    def calculate(self) -> float:
        if 0 >= len(self.labels) or 0 >= len(self.logits) \
            or len(self.labels) != len(self.logits):
            return 0
        
        total_labels = np.concatenate(self.labels, axis=0)
        total_logits = np.concatenate(self.logits, axis=0)
        accuracy = compute_final_score(total_labels, total_logits)

        return accuracy

    def calculate_by_group(self) -> float:
        if 0 >= len(self.labels) or 0 >= len(self.logits) or 0 >= len(self.uids) \
            or len(self.labels) != len(self.logits) \
            or len(self.labels) != len(self.uids):
            return 0
        
        total_uids = np.concatenate(self.uids, axis=0)
        total_group_uids = np.array([uid.split('_')[0] for uid in total_uids])

        group_indices = defaultdict(list)
        for index, group_uid in enumerate(total_group_uids):
            group_indices[group_uid].append(index)

        total_labels = np.concatenate(self.labels, axis=0)
        total_logits = np.concatenate(self.logits, axis=0)

        series_labels, series_logits = [], []

        for group_uid in np.unique(total_group_uids):
            series_labels.append(total_labels[group_indices[group_uid]].max(0))
            series_logits.append(total_logits[group_indices[group_uid]].max(0))
        
        series_labels = np.array(series_labels)
        series_logits = np.array(series_logits)

        accuracy = compute_final_score(series_labels, series_logits)

        return accuracy
    
    def save_predict(self):
        if 0 >= len(self.labels) or 0 >= len(self.logits) or 0 >= len(self.uids) \
            or len(self.labels) != len(self.logits) \
            or len(self.labels) != len(self.uids):
            return 0
        
        total_logits = np.concatenate(self.logits, axis=0)
        total_labels = np.concatenate(self.labels, axis=0)
        total_uids = np.concatenate(self.uids, axis=0)

        predict_save_folder = os.path.join(self.predict_save_path, f'{self.train_start_time_str}')
        os.makedirs(predict_save_folder, exist_ok=True)

        np.save(os.path.join(predict_save_folder, f'predict_{self.current_fold:02d}.npy'), total_logits)
        np.save(os.path.join(predict_save_folder, f'label_{self.current_fold:02d}.npy'), total_labels)
        np.save(os.path.join(predict_save_folder, f'uid_{self.current_fold:02d}.npy'), total_uids)


class ClassifierTrainer:
    def __init__(self
                #  , cfg: DictConfig
                 , fp16: bool
                 , grad_norm_clip_max: int
                 , max_epoch: int
                 , save_path: str
                 , predict_save_path: str
                 , optimizer: DictConfig
                 , lr_scheduler: DictConfig
                 , criterion: DictConfig
                 , ema: DictConfig
                 , device: torch.device|str
                 , data_manager: SeriesDataManager
                 , model: torch.nn.Module
                 ):
        self.max_epoch = max_epoch
        self.save_path = save_path
        self.predict_save_path = predict_save_path

        self.device = device
        self.grad_scaler = torch.GradScaler(enabled=fp16)
        self.grad_norm_clip_max = grad_norm_clip_max

        self.data_manager = data_manager
        self.train_dataloader = self.data_manager.get_train_dataloader()
        
        self.model = model.to(self.device)

        optim_type = pydoc.locate(optimizer.target_type)
        self.optimizer = optim_type(self.model.parameters(), **optimizer.kwargs)

        self.lr_scheduler_per_mini_batch = False
        self.lr_scheduler_step_period = 1
        steps_per_epoch: int = 1
        if 'step' in lr_scheduler and lr_scheduler.step.custom_enable:
            if lr_scheduler.step.per_mini_batch:
                self.lr_scheduler_per_mini_batch = True
                steps_per_epoch *= len(self.train_dataloader)
            self.lr_scheduler_step_period = lr_scheduler.step.period

            lr_scheduler_total_count = math.ceil(self.max_epoch * steps_per_epoch / self.lr_scheduler_step_period)
            lr_scheduler_total_count \
                = int((lr_scheduler_total_count * lr_scheduler.step.total_count.scale)
                      + lr_scheduler.step.total_count.bias)
            
            if 0 < len(lr_scheduler.step.arg_name):
                lr_scheduler.kwargs[lr_scheduler.step.arg_name] = lr_scheduler_total_count

        lr_scheduler_type = pydoc.locate(lr_scheduler.target_type)
        self.scheduler = lr_scheduler_type(optimizer=self.optimizer, **lr_scheduler.kwargs)

        self.criterion = criterion.bce.to(device)

        self.evaluator \
            = DicomSeriesEvaluator(self.device
                                   , self.data_manager.get_valid_dataloader()
                                   , self.data_manager.get_valid_sub_dataloader()
                                   , ClassifierAccuracyCalculator(
                                       predict_save_path=self.predict_save_path
                                       , current_fold=self.data_manager.current_fold
                                       ))
        
        self.model_ema: ModelEmaV3|None = None
        self.ema_update_start_step: int = 0
        if ema.enable:
            self.model_ema = ModelEmaV3(self.model, decay=ema.decay, device=self.device)
            self.ema_update_start_step = ema.update_start_step

        self.acc_calculator = ClassifierAccuracyCalculator(
            predict_save_path=self.predict_save_path
            , current_fold=self.data_manager.current_fold
            )
        self.train_start_time_str = time.strftime('%Y%m%d_%H%M%S')
        self.start_epoch = 0

    def train(self, model: torch.nn.Module):
        is_autocast_enabled = torch.is_autocast_enabled(str(self.device))
        autocast_dtype = torch.get_autocast_dtype(str(self.device))
        logger.info(f'is_autocast_enabled: {is_autocast_enabled}, autocast_dtype: {autocast_dtype}')
        
        # torch.autograd.set_detect_anomaly(True)

        self.train_losses = []
        self.valid_losses = []

        os.makedirs(self.save_path, exist_ok=True)

        model_save_folder = os.path.join(self.save_path, f'{type(self.model).__name__.lower()}_{self.train_start_time_str}')
        os.makedirs(model_save_folder, exist_ok=True)

        self.best_loss_checkpoint_name \
            = f'best_loss_checkpoint_{type(self.model).__name__.lower()}_{self.data_manager.current_fold:02d}_{self.train_start_time_str}.pth'
        self.best_acc_checkpoint_name \
            = f'best_acc_checkpoint_{type(self.model).__name__.lower()}_{self.data_manager.current_fold:02d}_{self.train_start_time_str}.pth'
        
        for epoch in range(self.start_epoch, self.max_epoch):
            model.train()

            train_loss, train_accuracy = self.train_one_epoch(model, epoch)

            valid_loss, valid_slide_accuracy, valid_accuracy = self.eval(model)

            self.train_losses.append(train_loss)
            self.valid_losses.append(valid_loss)
            
            logger.info(f'Fold {self.data_manager.current_fold+1}/{self.data_manager.num_fold}, Epoch {epoch+1}/{self.max_epoch}:')
            logger.info(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
            logger.info(f'lr: {"{:2e}".format(self.optimizer.param_groups[0]['lr'])}')
            logger.info(f'Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_accuracy:.4f}')
            logger.info(f'Validation Slide Accuracy: {valid_slide_accuracy:.4f}')

            wandb.log(
                {
                    "train_loss": train_loss,
                    "train_acc": train_accuracy,
                    "valid_loss": valid_loss,
                    "valid_acc": valid_accuracy,
                    "valid_slide_acc": valid_slide_accuracy,
                }
                , step=epoch+1)
            
            if self.evaluator.update_best_valid_loss(valid_loss):
                logger.info(f'Validation loss improved from {self.evaluator.prev_best_valid_loss:.4f} '
                      f'to {self.evaluator.best_valid_loss:.4f}.')
                logger.info('Save checkpoint.')
                torch.save(model.state_dict(), os.path.join(model_save_folder, self.best_loss_checkpoint_name))
            
            if self.evaluator.update_best_valid_acc(valid_accuracy):
                logger.info(f'Validation accuracy improved from {self.evaluator.prev_best_valid_acc:.4f} '
                      f'to {self.evaluator.best_valid_acc:.4f}.')
                logger.info('Save checkpoint.')
                torch.save(model.state_dict(), os.path.join(model_save_folder, self.best_acc_checkpoint_name))

    def eval(self, model: torch.nn.Module) -> tuple[float, float, float]:
        model_eval = model if self.model_ema is None else self.model_ema
        model_eval.eval()
        
        # Evaluate
        valid_loss, valid_slide_accuracy, valid_accuracy \
            = self.evaluator.evaluate(model_eval, self.criterion)

        return valid_loss, valid_slide_accuracy, valid_accuracy

    def train_one_epoch(self, model: torch.nn.Module, epoch: int) -> tuple[float, float]:
        train_loss = 0
        self.acc_calculator.reset()

        epoch_size = len(self.train_dataloader)
        optimizer_step_skip_batch_index = []
        grad_norm_nan_count = int(0)
        grad_norm_inf_count = int(0)

        for batch_index, batch_data in enumerate(tqdm(self.train_dataloader)):
            forward_step = batch_index + (epoch * epoch_size)

            images, labels, uids = batch_data
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            with torch.autocast(device_type=str(self.device)):
                pred = model(images)
                cost = self.criterion(pred, labels)
            
            self.grad_scaler.scale(cost).backward()

            grad_norm = None
            if self.grad_norm_clip_max > 0:
                # unscale gradients
                self.grad_scaler.unscale_(self.optimizer)
                # clip gradients
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.grad_norm_clip_max)
                if torch.logical_or(grad_norm.isnan(), grad_norm.isinf()):
                    optimizer_step_skip_batch_index.append(batch_index)
                    grad_norm_nan_count += int(1 if grad_norm.isnan() else 0)
                    grad_norm_inf_count += int(1 if grad_norm.isinf() else 0)

            # optimizer.step
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.optimizer.zero_grad()
            
            if self.lr_scheduler_per_mini_batch \
                and 0 == (forward_step + 1) % self.lr_scheduler_step_period:
                self.scheduler.step()

            if self.model_ema is not None and forward_step >= self.ema_update_start_step:
                self.model_ema.update(model)

            # min_scale = 128
            # if self.grad_scaler._scale < min_scale:
            #     self.grad_scaler._scale = torch.tensor(min_scale).to(self.grad_scaler._scale)
            
            # train_loss += cost.item()
            train_loss += (cost.item() - train_loss) * (1 / (batch_index + 1))
            
            self.acc_calculator.enqueue(labels, pred, uids)

            # if batch_index > 4:
            #     break
        
        # train_loss /= len(self.train_dataloader)
        train_accuracy = self.acc_calculator.calculate()

        if not self.lr_scheduler_per_mini_batch \
              and 0 == (epoch + 1) % self.lr_scheduler_step_period:
            self.scheduler.step()
        
        if 0 < len(optimizer_step_skip_batch_index):
            skip_step_str \
                = f'{len(optimizer_step_skip_batch_index) if 10 < len(optimizer_step_skip_batch_index)
                     else optimizer_step_skip_batch_index}'
            # scaler is going to skip optimizer.step() if grads are nan or inf
            # some updates are skipped anyway in the amp mode, but we can count for statistics
            logger.info(f'Skiped optimizer step, indices or count: {skip_step_str}')
            logger.info(f'nan_count: {grad_norm_nan_count}, inf_count: {grad_norm_inf_count}')

        return train_loss, train_accuracy
