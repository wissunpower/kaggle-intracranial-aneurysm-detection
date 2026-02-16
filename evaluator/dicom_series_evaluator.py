
from tqdm.auto import tqdm

import torch


class DicomSeriesEvaluator:
    def __init__(self
                 , device: torch.device|str
                 , valid_dataloader: torch.utils.data.DataLoader
                 , valid_sub_dataloader: torch.utils.data.DataLoader
                 , acc_calculator
                 ):
        self.device = device
        self.valid_dataloader = valid_dataloader
        self.valid_sub_dataloader = valid_sub_dataloader
        self.acc_calculator = acc_calculator
        
        self.best_valid_loss = float('inf')
        self.prev_best_valid_loss = float('inf')
        self.best_valid_acc = float(0)
        self.prev_best_valid_acc = float(0)
    
    def evaluate(self, model: torch.nn.Module, criterion: torch.nn.Module|None) \
        -> tuple[float, float, float]:
        # Validation
        valid_loss = 0
        self.acc_calculator.reset()
        
        model.eval()
        with torch.no_grad():
            for batch_index, batch_data in enumerate(tqdm(self.valid_dataloader)):
                images, labels, uids = batch_data
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                with torch.autocast(device_type=str(self.device)):
                    logits = model(images)
                    
                    if criterion is not None:
                        loss = criterion(logits, labels)
                        valid_loss += loss.item()
                
                self.acc_calculator.enqueue(labels, logits, uids)
                
                # if batch_index > 4:
                #     break
        
        valid_loss /= len(self.valid_dataloader)
        valid_slide_accuracy = self.acc_calculator.calculate()
        
        with torch.no_grad():
            for batch_index, batch_data in enumerate(tqdm(self.valid_sub_dataloader)):
                images, labels, uids = batch_data
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                with torch.autocast(device_type=str(self.device)):
                    logits = model(images)

                self.acc_calculator.enqueue(labels, logits, uids)
                
                # if batch_index > 4:
                #     break
        
        valid_accuracy = self.acc_calculator.calculate_by_group()

        self.acc_calculator.save_predict()

        return valid_loss, valid_slide_accuracy, valid_accuracy
    
    def update_best_valid_loss(self, new_loss: float) -> bool:
        if new_loss is not None and new_loss < self.best_valid_loss:
            self.prev_best_valid_loss = self.best_valid_loss
            self.best_valid_loss = new_loss
            return True
        
        return False
    
    def update_best_valid_acc(self, new_acc: float) -> bool:
        if new_acc is not None and new_acc > self.best_valid_acc:
            self.prev_best_valid_acc = self.best_valid_acc
            self.best_valid_acc = new_acc
            return True
        
        return False
