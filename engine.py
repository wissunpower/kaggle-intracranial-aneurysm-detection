
import argparse

import matplotlib.pyplot as plt

import torch

from dataset import *
from dataset.config import SeriesDataConfig
from evaluator import *
from utils.metrics import *
from utils.optim import lrfn


class DetectorTrainer:
    def __init__(self, args: argparse.Namespace, data_config: SeriesDataConfig, device: torch.device|str
                 , model: torch.nn.Module, fold_index: int):
        self.args = args
        self.device = device

        self.data_config = data_config
        self.fold_index = fold_index

        self.transform = build_transform(self.args)
        self.train_dataset = \
            build_dataset(self.args, self.data_config, self.fold_index, self.transform, is_train=True)
        self.train_dataloader = \
            torch.utils.data.DataLoader(self.train_dataset, self.args.batch_size
                                        , shuffle=True, collate_fn=self.train_dataset.collate_fn
                                        , drop_last=True)
        
        self.clip_grad = 35
        self.grad_scaler = torch.GradScaler(enabled=self.args.fp16)

        self.criterion = torch.nn.BCEWithLogitsLoss().to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, lr_lambda=lrfn)
        # self.scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer=self.optimizer, factor=0.00001
        #                                                      , total_iters=self.args.max_epoch)

        self.evaluator = build_evaluator(self.args, self.data_config, self.device, self.fold_index)

        self.start_epoch = 0

    def train(self, model: torch.nn.Module):
        is_autocast_enabled = torch.is_autocast_enabled(str(self.device))
        autocast_dtype = torch.get_autocast_dtype(str(self.device))
        print(f'is_autocast_enabled: {is_autocast_enabled}, autocast_dtype: {autocast_dtype}')
        
        # torch.autograd.set_detect_anomaly(True)

        self.train_losses = []
        self.valid_losses = []

        os.makedirs(self.args.save_folder, exist_ok=True)

        model_save_folder = os.path.join(self.args.save_folder, f'{self.args.model}_{self.data_config.save_start_time_str}')
        os.makedirs(model_save_folder, exist_ok=True)

        self.best_checkpoint_name \
            = f'best_checkpoint_{self.args.model}_{self.fold_index:02d}_{self.data_config.save_start_time_str}.pth'
        
        for epoch in range(self.start_epoch, self.args.max_epoch):
            model.train()

            train_loss, train_accuracy = self.train_one_epoch(model, epoch)

            valid_loss, valid_accuracy = self.eval(model)

            self.train_losses.append(train_loss)
            self.valid_losses.append(valid_loss)
            
            print(f'Fold {self.fold_index+1}/{self.args.num_fold}, Epoch {epoch+1}/{self.args.max_epoch}:')
            print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
            print(f'Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_accuracy:.4f}')
            
            if self.evaluator.update_best_valid_loss(valid_loss):
                print(f'Validation loss improved from {self.evaluator.prev_best_valid_loss:.4f} '
                      f'to {self.evaluator.best_valid_loss:.4f}.')
                print('Save checkpoint.')
                torch.save(model.state_dict(), os.path.join(model_save_folder, self.best_checkpoint_name))

    def eval(self, model: torch.nn.Module) -> tuple[float, float]:
        model.eval()
        
        # Evaluate
        valid_loss, valid_accuracy = self.evaluator.evaluate(model, self.criterion)

        return valid_loss, valid_accuracy

    def train_one_epoch(self, model: torch.nn.Module, epoch: int) -> tuple[float, float]:
        train_loss = 0
        train_correct = 0
        train_total = 0

        optimizer_step_skip_start_batch_index = int(-1)
        grad_norm_isnan = False
        grad_norm_isinf = False

        for batch_index, batch_data in enumerate(tqdm(self.train_dataloader)):
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

            # min_scale = 128
            # if self.grad_scaler._scale < min_scale:
            #     self.grad_scaler._scale = torch.tensor(min_scale).to(self.grad_scaler._scale)
            
            train_loss += cost.item()
            current_accuracies = weighted_multilabel_auc_for_multiset(
                                labels.detach().cpu().numpy(),
                                pred.sigmoid().detach().cpu().numpy(),
                                self.data_config.label_auc_weights)
            train_correct += np.sum(current_accuracies)
            train_total += len(current_accuracies)

            # if batch_index > 4:
            #     break
        
        train_accuracy = train_correct / train_total
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


def build_trainer(args: argparse.Namespace, data_config: SeriesDataConfig, device: torch.device|str
                  , model: torch.nn.Module, fold_index: int=0) \
    -> DetectorTrainer:
    return DetectorTrainer(args, data_config, device, model, fold_index)
