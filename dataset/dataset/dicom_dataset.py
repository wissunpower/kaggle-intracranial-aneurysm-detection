
import os
import numpy as np
import pandas as pd

import torch


class DICOMDataset(torch.utils.data.Dataset):
    def __init__(self
                 , data_root_dir: str
                 , metadata_df: pd.DataFrame
                 , localizer_df: pd.DataFrame
                 , roi_crop_info: dict|None
                 , num_label_classes: int
                 , base_transform=None
                 , aug_transform=None
        ):
        self.metadata_df = metadata_df
        self.localizer_df = localizer_df
        self.roi_crop_info = roi_crop_info
        self.num_label_classes = num_label_classes
        self.base_transform = base_transform
        self.aug_transform = aug_transform

        self.col_to_idx \
            = {x: i for i, x in enumerate(self.metadata_df.columns[6:6+self.num_label_classes-1])}
        
        self.raw_data_root_path = data_root_dir
        
        if not os.path.exists(self.raw_data_root_path):
            print(f'Not found raw data(npz image files) path'
                  f', preprocess data folder path: {self.raw_data_root_path}')
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        image, multi_label, uid = self.get_raw_data(index)

        if self.base_transform is not None:
            image = self.base_transform(image)

        if self.aug_transform is not None:
            image = self.aug_transform(image)
        
        image = torch.from_numpy(image).contiguous().float()
        label = torch.from_numpy(multi_label).float()

        return image, label, uid
    
    def get_raw_data(self, index: int) -> tuple[np.ndarray, np.ndarray, str]:
        index = index % len(self.metadata_df)

        row = self.metadata_df.iloc[index]

        raw_image_file_path \
            = os.path.join(self.raw_data_root_path, f'{row.SeriesUID}_I_{row.InstanceNumber}.npy')
        file_stream = np.load(raw_image_file_path).astype(np.float64)

        try:
            prev_raw_image_file_path \
                = os.path.join(self.raw_data_root_path, f'{row.SeriesUID}_I_{row.InstanceNumber-2}.npy')
            prev_file_stream = np.load(prev_raw_image_file_path).astype(np.float64)
        except:
            prev_file_stream = file_stream

        try:
            next_raw_image_file_path \
                = os.path.join(self.raw_data_root_path, f'{row.SeriesUID}_I_{row.InstanceNumber+2}.npy')
            next_file_stream = np.load(next_raw_image_file_path).astype(np.float64)
        except:
            next_file_stream = file_stream

        image = np.stack([prev_file_stream, file_stream, next_file_stream])

        if self.roi_crop_info is not None:
            height, width = image.shape[-2:]
            x1, x2, y1, y2 = self.roi_crop_info[row.SeriesUID]
            image = image[:
                          , int(y1 * height * 0.9):int(y2 * height * 1.1)
                          , int(x1 * width * 0.9):int(x2 * width * 1.1)]

        if image.max() != image.min():
            image = (image - image.min()) / (image.max() - image.min())
        # else:
        #     print(f'uid: {row.SeriesUID}_I_{row.InstanceNumber}, min: {image.min()}, max: {image.max()}, gap: {(image.max() - image.min())}')

        localizer_label_rows = self.localizer_df[self.localizer_df.SOPInstanceUID == row.InstanceUID]
        if len(localizer_label_rows) > 0:
            multi_label = np.zeros((self.num_label_classes,))
            for _, localizer_label_row in localizer_label_rows.iterrows():
                loc = localizer_label_row.location.replace(' ', '_').lower()
                multi_label[self.col_to_idx[loc]] = 1.
            multi_label[-1] = row.aneurysm_present
        else:
            multi_label = row[6 : 6 + self.num_label_classes].to_numpy()

        return image.astype(np.float32) \
            , multi_label.astype(np.float32) \
            , f'{row.SeriesUID}_I_{row.InstanceNumber}'
    
    def __len__(self) -> int:
        return len(self.metadata_df)
