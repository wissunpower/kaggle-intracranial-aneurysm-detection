
import os
import numpy as np
import pandas as pd

import torch

from dataset.transform.series_transform import SeriesTransform


class DICOMDataset(torch.utils.data.Dataset):
    def __init__(self
                 , data_root_dir: str
                 , input_data_folder_name: str
                 , metadata_df: pd.DataFrame
                 , df_indices: np.ndarray
                 , transform: SeriesTransform|None=None
        ):
        self.metadata_df = metadata_df
        self.df_indices = df_indices
        self.transform = transform

        self.scan_df()
        
        self.raw_data_root_path = os.path.join(data_root_dir, input_data_folder_name)
        
        if not os.path.exists(self.raw_data_root_path):
            print(f'Not found raw data(npz image files) path'
                  f', preprocess data folder path: {self.raw_data_root_path}')
    
    def scan_df(self):
        if len(self.metadata_df) <= self.df_indices.max():
            print(f'Invalid label df indices'
                  f', label df size: {self.metadata_df}, max df index: {self.df_indices.max()}')
            return
        
        self.label_infos = []

        for df_index in self.df_indices:
            current_row = self.metadata_df.values[df_index]
            series_instance_uid = current_row[0]
            multi_label = current_row[4:].astype(np.int8)
            self.label_infos.append((df_index, series_instance_uid, multi_label))
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, multi_label = self.get_raw_data(index)

        image = image.astype(np.float32)

        if self.transform is not None:
            image = self.transform(image)
        
        image = torch.from_numpy(image).contiguous().float() / 255.
        label = torch.from_numpy(multi_label).float()

        return image, label
    
    def get_raw_data(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        index = index % len(self.label_infos)

        df_index, series_instance_uid, multi_label = self.label_infos[index]

        raw_image_file_path = os.path.join(self.raw_data_root_path, series_instance_uid + '_origsampled.npy')
        file_stream = np.load(raw_image_file_path)
        # image = np.transpose(file_stream['arr_0'], (1, 2, 0))  # dim: [D, H, W] ->  [H, W, D]
        # image = file_stream['arr_0']  # dim: [D, H, W]
        image = file_stream

        return image, multi_label
    
    def __len__(self) -> int:
        return len(self.label_infos)
    
    def collate_fn(self, batch):
        images, labels = list(zip(*batch))

        if self.transform is not None:
            self.transform.step_by_batch()
        
        images = torch.stack(images)
        labels = torch.stack(labels)

        return images, labels
