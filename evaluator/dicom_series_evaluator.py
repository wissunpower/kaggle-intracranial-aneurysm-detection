
import numpy as np
from omegaconf import DictConfig
from tqdm.auto import tqdm

import torch

from dataset import build_transform, build_dataset
from dataset.config import SeriesDataConfig
from utils.metrics import compute_final_score


class DicomSeriesEvaluator:
    def __init__(self, cfg: DictConfig, data_config: SeriesDataConfig, device: torch.device|str, fold_index: int=0):
        self.data_config = data_config
        self.device = device
        
        transform = build_transform(cfg.data.nifti_transform)
        self.valid_dataset = build_dataset(cfg, self.data_config, fold_index, transform)
        self.valid_dataloader = torch.utils.data.DataLoader(self.valid_dataset, cfg.data.batch_size)
        
        self.best_valid_loss = float('inf')
        self.prev_best_valid_loss = float('inf')
    
    def evaluate(self, model: torch.nn.Module, criterion: torch.nn.Module|None) -> tuple[float, float]:
        # Validation
        valid_loss = 0
        valid_logits: list[np.ndarray] = []
        valid_labels: list[np.ndarray] = []
        
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
                
                valid_labels.append(labels.detach().cpu().numpy())
                valid_logits.append(logits.sigmoid().detach().cpu().numpy())
                
                # if batch_index > 4:
                #     break
        
        valid_accuracy: float = 0
        if 0 < len(valid_labels) and 0 < len(valid_logits):
            valid_y_true = np.concatenate(valid_labels, axis=0)
            valid_y_score = np.concatenate(valid_logits, axis=0)
            valid_accuracy = compute_final_score(valid_y_true, valid_y_score)
        valid_loss /= len(self.valid_dataloader)
        
        return valid_loss, valid_accuracy
    
    def update_best_valid_loss(self, new_loss: float) -> bool:
        if new_loss is not None and new_loss < self.best_valid_loss:
            self.prev_best_valid_loss = self.best_valid_loss
            self.best_valid_loss = new_loss
            return True
        
        return False
