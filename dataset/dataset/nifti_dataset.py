
import os
import numpy as np
import pandas as pd
import glob

import torch
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIOWithReorient

from utils.log import logger
from dataset import NiftiTransform


class NiftiDataset(torch.utils.data.Dataset):
    def __init__(self
                 , niix_path: str
                 , df_indices: np.ndarray
                 , label_df: pd.DataFrame
                 , transform: NiftiTransform|None=None):
        self.label_df = label_df
        self.df_indices = df_indices
        self.transform = transform

        self.scan_df()
    
        self.niix_path = niix_path
        self.nifti_io_helper = SimpleITKIOWithReorient()
    
    def scan_df(self):
        if len(self.label_df) <= self.df_indices.max():
            logger.error(f'Invalid label df indices'
                  f', label df size: {self.label_df}, max df index: {self.df_indices.max()}')
            raise IndexError
        
        self.label_infos = []

        for df_index in self.df_indices:
            current_row = self.label_df.values[df_index]
            series_instance_uid = current_row[0]
            multi_label = current_row[4:].astype(np.int8)
            self.label_infos.append((df_index, series_instance_uid, multi_label))
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        volume, multi_label = self.get_raw_data(index)

        if self.transform is not None:
            volume = self.transform(volume)
        
        volume = torch.from_numpy(volume).squeeze(0).contiguous().float() / 255.
        label = torch.from_numpy(multi_label).float()

        return volume, label
    
    def get_raw_data(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        index = index % len(self.label_infos)

        _, series_instance_uid, multi_label = self.label_infos[index]

        nifti_file_name_pattern = os.path.join(self.niix_path, series_instance_uid, '*.nii.gz')
        nifti_files = glob.glob(nifti_file_name_pattern)
        if 1 > len(nifti_files):
            logger.error(f'Not found nifti file'
                  f', nifti file name pattern: {nifti_file_name_pattern}')
            raise FileExistsError()
        
        volume, _ = self.nifti_io_helper.read_images([nifti_files[0]])

        return volume, multi_label
    
    def __len__(self) -> int:
        return len(self.label_infos)
