
import os
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tqdm.auto import tqdm

import torch
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIOWithReorient

from utils.log import logger
from dataset import NiftiDataset, VesselSegmentDataset

class VesselSegmentDataManager:
    def __init__(self
                 , volumes_path: str
                 , labels_path: str
                 , num_fold: int
                 , batch_size: int
                 , volume_input_size: list[int]
                 , num_mask_classes: int
                 , transform: NiftiDataset
                 , train_dataset: VesselSegmentDataset
                 , valid_dataset: VesselSegmentDataset
                 , data_common_cfg: DictConfig
        ):
        self.data_label_df = pd.read_csv(data_common_cfg.label_file_path)
        self.volumes_path = volumes_path
        self.labels_path = labels_path
 
        '''
            self.num_classes is 'Aneurysm Present' and 13 types of vessel parts = 14
            self.num_mask_classes is back-ground and 13 types of vessel parts(fore-ground) = 14
        '''
        self.label_class_names = data_common_cfg.label_class_names
        self.num_classes = len(self.label_class_names)
        # except 'Aneurysm Present' only vessel part
        self.vessel_classes = self.num_classes - 1
        # include back-ground
        self.num_mask_classes = num_mask_classes

        self.num_fold = num_fold
        self.batch_size = batch_size

        self.transform = transform
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

        self.nifti_io_helper = SimpleITKIOWithReorient()
    
    def confirm_raw_data(self):
        volume_files = sorted([i for i in os.listdir(self.volumes_path) if i.endswith('.nii.gz')])
        label_files = sorted([i for i in os.listdir(self.labels_path) if i.endswith('.nii.gz')])

        self.cached_data = list[dict]()
        # back ground + number of vessel classes
        self.count_by_vessel_class = np.zeros((1 + self.vessel_classes,), int)

        for _, file_names in enumerate(tqdm(zip(volume_files, label_files), desc='confirm raw data')):
            volume_file_name, label_file_name = file_names
            uid = volume_file_name.split('_')[0]
            if uid != label_file_name[:-7]:
                logger.error(f'not matched volume and label file name.'
                  f', volume_file_name: {volume_file_name}, label_file_name: {label_file_name}')
                continue

            # volume, volume_prop = self.nifti_io_helper.read_images(
            #     [os.path.join(self.volumes_path, volume_file_name)], orientation="RAS")
            # label, label_prop = self.nifti_io_helper.read_images(
            #     [os.path.join(self.labels_path, label_file_name)], orientation="RAS")
            
            # mask_labels = np.unique(label).astype(int)
            # self.count_by_vessel_class[mask_labels] += 1

            self.cached_data.append({
                'uid': uid,
                # 'volume': volume,
                # 'volume_prop': volume_prop,
                # 'label': label,
                # 'label_prop': label_prop,
                # 'mask_labels': mask_labels,
            })

    def prepare_data(self):
        self.confirm_raw_data()
        self.train_data_indices, self.valid_data_indices \
            = self.split_data_index_with_fold(self.num_fold) if 1 < self.num_fold \
                else self.split_data_index()

        train_uids = [cached_data.get('uid') for cached_data in self.cached_data
                                                if 'uid' in cached_data]
        self.train_dataset.initialize(train_uids, self.transform)
        self.train_dataloader = \
            torch.utils.data.DataLoader(self.train_dataset, self.batch_size
                                        , shuffle=True, drop_last=True)

        valid_uids = [cached_data.get('uid') for cached_data in self.cached_data
                                                if 'uid' in cached_data]
        self.valid_dataset.initialize(valid_uids, self.transform)
        self.valid_dataloader = \
            torch.utils.data.DataLoader(self.valid_dataset, self.batch_size
                                        , shuffle=True, drop_last=True)

    def split_data_index_with_fold(self, num_fold: int=1) -> tuple[list[np.ndarray], list[np.ndarray]]:
        num_data = len(self.cached_data)
        total_indices = np.arange(num_data)

        np.random.shuffle(total_indices)
        
        return [np.array(total_indices)], [np.array(total_indices)]

    def split_data_index(self, valid_data_rate: float = 0.2) -> tuple[list[np.ndarray], list[np.ndarray]]:
        num_data = len(self.cached_data)
        total_indices = np.arange(num_data)

        np.random.shuffle(total_indices)
        
        return [np.array(total_indices)], [np.array(total_indices)]
    
    def get_train_dataloader(self) -> torch.utils.data.DataLoader:
        return self.train_dataloader
    
    def get_valid_dataloader(self) -> torch.utils.data.DataLoader:
        return self.valid_dataloader
