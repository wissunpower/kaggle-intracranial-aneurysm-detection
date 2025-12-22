
import numpy as np
from tqdm.auto import tqdm

import torch

from utils.metrics import compute_final_score


class VesselSegmentEvaluator:
    def __init__(self
                 , device: torch.device|str
                 , valid_dataloader: torch.utils.data.DataLoader
                 ):
        self.device = device
        self.valid_dataloader = valid_dataloader
        
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
                volumes, labels = batch_data
                volumes = volumes.to(self.device)
                labels = labels.to(self.device)
                
                logits = model(volumes)
                
                if criterion is not None:
                    loss = criterion(logits, labels)
                    valid_loss += loss.item()
                
                num_classes = labels.shape[1]
                metrics_depth = slice(int(labels.shape[-3] / 2) - int(labels.shape[-3] / 6)
                                      , int(labels.shape[-3] / 2) + int(labels.shape[-3] / 6))
                metrics_height = slice(int(labels.shape[-2] / 2) - int(labels.shape[-2] / 6)
                                       , int(labels.shape[-2] / 2) + int(labels.shape[-2] / 6))
                metrics_width = slice(int(labels.shape[-1] / 2) - int(labels.shape[-1] / 6)
                                      , int(labels.shape[-1] / 2) + int(labels.shape[-1] / 6))

                valid_labels.append(labels.detach().cpu().numpy()
                                    .transpose(0, 2, 3, 4, 1)[:, metrics_depth, metrics_height, metrics_width, :]
                                    .reshape(-1, num_classes))
                valid_logits.append(logits.sigmoid().detach().cpu().numpy()
                                    .transpose(0, 2, 3, 4, 1)[:, metrics_depth, metrics_height, metrics_width, :]
                                    .reshape(-1, num_classes))
                
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
