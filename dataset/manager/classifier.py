
import os
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tqdm.auto import tqdm

import torch

from dataset import DICOMDataset


ID_COL = 'SeriesInstanceUID'

LABEL_COLS = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present',
]

NUM_CLASSES = len(LABEL_COLS)

# All tags (other than PixelData and SeriesInstanceUID) that may be in a test set dcm file
DICOM_TAG_ALLOWLIST = [
    'BitsAllocated',
    'BitsStored',
    'Columns',
    'FrameOfReferenceUID',
    'HighBit',
    'ImageOrientationPatient',
    'ImagePositionPatient',
    'InstanceNumber',
    'Modality',
    'PatientID',
    'PhotometricInterpretation',
    'PixelRepresentation',
    'PixelSpacing',
    'PlanarConfiguration',
    'RescaleIntercept',
    'RescaleSlope',
    'RescaleType',
    'Rows',
    'SOPClassUID',
    'SOPInstanceUID',
    'SamplesPerPixel',
    'SliceThickness',
    'SpacingBetweenSlices',
    'StudyInstanceUID',
    'TransferSyntaxUID',
]


class SeriesDataManager:
    def __init__(self
                 , input_data_folder_name: str
                 , series_slide_fileset_path: str
                 , series_slide_metainfo_file_name: str
                 , num_fold: int
                 , current_fold: int
                 , batch_size: int
                 , num_series_slide: int
                 , nifti_transform
                 , data_common_cfg: DictConfig
        ):
        self.data_root_path = data_common_cfg.data_root_path
        self.series_slide_fileset_path = series_slide_fileset_path
        self.data_label_df \
            = pd.read_csv(os.path.join(self.data_root_path, data_common_cfg.label_file_name))
        self.data_localizer_df \
            = pd.read_csv(os.path.join(self.data_root_path, data_common_cfg.localizer_label_file_name))
        self.series_slide_metainfo_df \
            = pd.read_csv(os.path.join(self.data_root_path, series_slide_metainfo_file_name))
        self.num_label_classes = data_common_cfg.num_classes

        self.input_data_folder_name = input_data_folder_name
        self.num_fold = num_fold
        self.current_fold = int(np.clip(current_fold, 0, self.num_fold - 1))
        self.batch_size = batch_size
        self.num_series_slide = num_series_slide
        self.transform = nifti_transform

        self.column_start_index = 4

        self.init_label_weight_priority()

        self.init_data_indices()

    def init_label_weight_priority(self):
        data_label_np = self.data_label_df.to_numpy()

        count_map = []
        
        for index in range(14):
            column_index = self.column_start_index + index
            total_count = np.sum(data_label_np[:, column_index] != 0)
            count_map.append([index, total_count])
        
        count_map.sort(key=lambda elem: elem[1])

        self.label_weight_priority = [elem[0] for elem in count_map]
    
    def init_data_indices(self):
        if 1 < self.num_fold:
            self.indices_folds = self.split_fold_data_index()
        
        self.train_data_default_indices, self.valid_data_default_indices = self.split_data_index()
    
    def get_train_data_indices(self, fold_index:int|None = None) -> np.ndarray:
        if 1 >= self.num_fold:
            return self.train_data_default_indices
        
        if fold_index is None:
            fold_index = self.current_fold
        
        fold_index = int(np.clip(fold_index, 0, self.num_fold - 1))
        
        return np.concatenate([indices for index, indices in enumerate(self.indices_folds)
                               if fold_index != index])
    
    def get_valid_data_indices(self, fold_index:int|None = None) -> np.ndarray:
        if 1 >= self.num_fold:
            return self.valid_data_default_indices
        
        if fold_index is None:
            fold_index = self.current_fold
        
        fold_index = int(np.clip(fold_index, 0, self.num_fold - 1))
        
        return np.array(self.indices_folds[fold_index])
    
    def build_dataset_metaInfo(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_indices = self.get_train_data_indices()
        valid_indices = self.get_valid_data_indices()

        train_data = self.data_label_df.iloc[train_indices]
        valid_data = self.data_label_df.iloc[valid_indices]

        train_sample = []
        valid_sample = []
        
        localizer_df_group = self.data_localizer_df.groupby('SeriesInstanceUID')

        for _, group_pair in tqdm(enumerate(localizer_df_group)):
            series_uid, group_df = group_pair
            
            if series_uid in train_data.SeriesInstanceUID.values.tolist():
                for _, slide_df in group_df.iterrows():
                    train_sample.append(
                        self.series_slide_metainfo_df[
                            self.series_slide_metainfo_df.InstanceUID == slide_df.SOPInstanceUID
                            ])
            
            if series_uid in valid_data.SeriesInstanceUID.values.tolist():
                for _, slide_df in group_df.iterrows():
                    valid_sample.append(
                        self.series_slide_metainfo_df[
                            self.series_slide_metainfo_df.InstanceUID == slide_df.SOPInstanceUID
                            ])
        
        metainfo_df_group = self.series_slide_metainfo_df.groupby('SeriesUID')

        for _, group_pair in enumerate(metainfo_df_group):
            series_uid, group_df = group_pair
            
            target_index_start = 0
            target_index_end = len(group_df) - 1
            if len(group_df) * 0.7 >= self.num_series_slide:
                margin = len(group_df) * 0.15
                target_index_start += margin
                target_index_end -= margin
            target_rows =\
                group_df.iloc[
                    np.linspace(target_index_start, target_index_end, self.num_series_slide).astype(int)
                ]

            # In train data case, include only negative data
            # positive data included above
            if series_uid in train_data.SeriesInstanceUID.values.tolist() \
                and 0 == group_df.aneurysm_present.values.sum():
                train_sample.append(target_rows)
            
            if series_uid in valid_data.SeriesInstanceUID.values.tolist() \
                and 0 == group_df.aneurysm_present.values.sum():
                valid_sample.append(target_rows)

        train_slide_data = pd.concat(train_sample).reset_index(drop=True)
        valid_slide_data = pd.concat(valid_sample).reset_index(drop=True)

        return train_slide_data, valid_slide_data
    
    def build_dataloader(self, fold_index:int|None = None):
        if fold_index is not None:
            self.current_fold = int(np.clip(fold_index, 0, self.num_fold - 1))

        train_slide_data, valid_slide_data = self.build_dataset_metaInfo()
 
        self.train_dataset = DICOMDataset(data_root_dir=self.series_slide_fileset_path
                                          , metadata_df=train_slide_data
                                          , localizer_df=self.data_localizer_df
                                          , num_label_classes=self.num_label_classes
                                          , transform=self.transform
                                          )
        self.train_dataloader = \
            torch.utils.data.DataLoader(self.train_dataset, self.batch_size
                                        , shuffle=True, drop_last=True)

        self.valid_dataset = DICOMDataset(data_root_dir=self.series_slide_fileset_path
                                          , metadata_df=valid_slide_data
                                          , localizer_df=self.data_localizer_df
                                          , num_label_classes=self.num_label_classes
                                          , transform=self.transform
                                          )
        self.valid_dataloader = torch.utils.data.DataLoader(self.valid_dataset, self.batch_size)

    def get_excepted_indices(self) -> np.ndarray:
        indices = list[int]()

        return np.array(indices)

    def split_fold_data_index(self) -> list[list[int]]:
        num_data = len(self.data_label_df)

        data_label_np = self.data_label_df.to_numpy()
        total_indices = np.arange(num_data)

        excepted_indices = self.get_excepted_indices()
        selected_mask = np.full(num_data, False)
        if 0 < len(excepted_indices):
            selected_mask[excepted_indices] = True

        indices_folds = [[] for _ in range(self.num_fold)]

        for label_index in self.label_weight_priority:
            positive_indices = np.where(data_label_np[:, self.column_start_index + label_index] != 0)[0]
            positive_indices = np.setdiff1d(positive_indices, total_indices[selected_mask])
            num_positive = len(positive_indices)

            if 0 >= num_positive:
                continue

            for fold_index in range(self.num_fold):
                pii = np.where(np.arange(num_positive) % self.num_fold == fold_index)[0]
                indices_folds[fold_index].extend(positive_indices[pii])

            selected_mask[positive_indices] = True
        
        remain_indices = total_indices[selected_mask != True]
        num_remain = len(remain_indices)

        if 0 < num_remain:
            for fold_index in range(self.num_fold):
                rii = np.where(np.arange(num_remain) % self.num_fold == fold_index)[0]
                indices_folds[fold_index].extend(remain_indices[rii])

        total_count = 0
        for indices_fold in indices_folds:
            assert len(indices_fold) == len(np.unique(indices_fold)), 'Duplicate train index exists.'
            total_count += len(np.unique(indices_fold))
        
        assert total_count == (num_data - len(excepted_indices)) \
            , 'Missing index exists.'
        
        return indices_folds

    def split_data_index(self, valid_data_rate: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
        num_data = len(self.data_label_df)
        valid_data_rate = np.clip(valid_data_rate, 0.0, 1.0)

        data_label_np = self.data_label_df.to_numpy()
        total_indices = np.arange(num_data)

        excepted_indices = self.get_excepted_indices()
        selected_mask = np.full(num_data, False)
        if 0 < len(excepted_indices):
            selected_mask[excepted_indices] = True

        train_indices = []
        valid_indices = []

        for label_index in self.label_weight_priority:
            positive_indices = np.where(data_label_np[:, self.column_start_index + label_index] != 0)[0]
            positive_indices = np.setdiff1d(positive_indices, total_indices[selected_mask])
            num_valid = int(len(positive_indices) * valid_data_rate)

            if 0 >= num_valid:
                continue
            
            train_indices.extend(positive_indices[num_valid:])
            valid_indices.extend(positive_indices[:num_valid])

            selected_mask[positive_indices] = True
        
        remain_indices = total_indices[selected_mask != True]
        num_valid = max(int(len(remain_indices) * valid_data_rate), 1)

        train_indices.extend(remain_indices[num_valid:])
        valid_indices.extend(remain_indices[:num_valid])

        np.random.shuffle(train_indices)
        np.random.shuffle(valid_indices)

        assert len(train_indices) == len(np.unique(train_indices)), 'Duplicate train index exists.'
        assert len(valid_indices) == len(np.unique(valid_indices)), 'Duplicate valid index exists.'
        assert len(valid_indices) + len(train_indices) == (num_data - len(excepted_indices)) \
            , 'Missing index exists.'
        
        return np.array(train_indices), np.array(valid_indices)
    
    def get_train_dataloader(self) -> torch.utils.data.DataLoader:
        return self.train_dataloader
    
    def get_valid_dataloader(self) -> torch.utils.data.DataLoader:
        return self.valid_dataloader
