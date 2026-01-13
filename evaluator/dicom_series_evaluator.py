
from tqdm.auto import tqdm

import torch


class DicomSeriesEvaluator:
    def __init__(self
                 , device: torch.device|str
                 , valid_dataloader: torch.utils.data.DataLoader
                 , acc_calculator
                 ):
        self.device = device
        self.valid_dataloader = valid_dataloader
        self.acc_calculator = acc_calculator
        
        self.best_valid_loss = float('inf')
        self.prev_best_valid_loss = float('inf')
    
    def evaluate(self, model: torch.nn.Module, criterion: torch.nn.Module|None) -> tuple[float, float]:
        # Validation
        valid_loss = 0
        self.acc_calculator.reset()
        
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
                
                self.acc_calculator.enqueue(labels, logits)
                
                # if batch_index > 4:
                #     break
        
        valid_loss /= len(self.valid_dataloader)
        valid_accuracy = self.acc_calculator.calculate()
        
        return valid_loss, valid_accuracy
    
    def update_best_valid_loss(self, new_loss: float) -> bool:
        if new_loss is not None and new_loss < self.best_valid_loss:
            self.prev_best_valid_loss = self.best_valid_loss
            self.best_valid_loss = new_loss
            return True
        
        return False
