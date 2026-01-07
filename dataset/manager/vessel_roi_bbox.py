
import numpy as np
import pandas as pd

import torch


class VesselROIBBoxDataManager:
    def __init__(self
                 , metadata_file_path: str
                 , num_fold: int
                 , batch_size: int
                 , num_series_slide: int
                 , depth_size: int
                 , image_size: list[int]
                 , train_dataset
                 , valid_dataset
        ):
        self.metadata_df = pd.read_csv(metadata_file_path)

        self.num_fold = num_fold
        self.batch_size = batch_size

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

    def prepare_data(self):
        self.train_data_indices, self.valid_data_indices = self.split_data_index()

        self.train_dataset.initialize(self.metadata_df.iloc[self.train_data_indices])
        self.train_dataloader = \
            torch.utils.data.DataLoader(self.train_dataset, self.batch_size
                                        , shuffle=True, drop_last=True)

        self.valid_dataset.initialize(self.metadata_df.iloc[self.valid_data_indices])
        self.valid_dataloader = \
            torch.utils.data.DataLoader(self.valid_dataset, self.batch_size
                                        , shuffle=True, drop_last=True)

    def split_data_index(self, valid_data_rate: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
        num_data = len(self.metadata_df)
        valid_data_rate = np.clip(valid_data_rate, 0.0, 1.0)

        total_indices = np.arange(num_data)
        num_valid = int(max(num_data * valid_data_rate, 1))

        np.random.shuffle(total_indices)
        
        return np.array(total_indices[num_valid:]), np.array(total_indices[:num_valid])
    
    def get_train_dataloader(self) -> torch.utils.data.DataLoader:
        return self.train_dataloader
    
    def get_valid_dataloader(self) -> torch.utils.data.DataLoader:
        return self.valid_dataloader
