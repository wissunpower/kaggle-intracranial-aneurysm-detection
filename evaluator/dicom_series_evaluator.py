
import numpy as np
import argparse
from tqdm.auto import tqdm

import torch

from dataset import build_dataset
from dataset.config import SeriesDataConfig
from utils.metrics import weighted_multilabel_auc_for_multiset


class DicomSeriesEvaluator:
    def __init__(self, args: argparse.Namespace, data_config: SeriesDataConfig, device: torch.device|str, fold_index: int=0):
        self.args = args
        self.data_config = data_config
        self.device = device
        
        self.valid_dataset = build_dataset(self.args, self.data_config, fold_index)
        self.valid_dataloader = torch.utils.data.DataLoader(self.valid_dataset, self.args.batch_size)
        
        self.best_valid_loss = float('inf')
        self.prev_best_valid_loss = float('inf')
    
    def evaluate(self, model: torch.nn.Module, criterion: torch.nn.Module|None) -> tuple[float, float]:
        # Validation
        valid_loss = 0
        valid_correct = 0
        valid_total = 0
        
        model.eval()
        with torch.no_grad():
            for batch_index, batch_data in enumerate(tqdm(self.valid_dataloader)):
                images, labels = batch_data
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                logits = model(images)
                
                if criterion is not None:
                    loss = criterion(logits, labels)
                    valid_loss += loss.item()
                
                current_accuracies = weighted_multilabel_auc_for_multiset(
                                    labels.detach().cpu().numpy(),
                                    logits.sigmoid().detach().cpu().numpy(),
                                    self.data_config.label_auc_weights)
                valid_correct += np.sum(current_accuracies)
                valid_total += len(current_accuracies)
                
                # if batch_index > 4:
                #     break
        
        valid_accuracy = valid_correct / valid_total
        valid_loss /= len(self.valid_dataloader)
        
        return valid_loss, valid_accuracy
    
    def update_best_valid_loss(self, new_loss: float) -> bool:
        if new_loss is not None and new_loss < self.best_valid_loss:
            self.prev_best_valid_loss = self.best_valid_loss
            self.best_valid_loss = new_loss
            return True
        
        return False
