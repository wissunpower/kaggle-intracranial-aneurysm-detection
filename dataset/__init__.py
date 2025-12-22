
import os
import argparse
import numpy as np
import pandas as pd
import glob

from omegaconf import DictConfig
import hydra

from tqdm.auto import tqdm

from dataset.config import SeriesDataConfig
from dataset.transform import SeriesTransform, NiftiTransform
from dataset.dataset import (
    DICOMDataset,
    NiftiDataset,
    VesselSegmentDataset,
)
from dataset.preprocessor import DICOMPreprocessor

from dataset.manager import VesselSegmentDataManager


def build_dataset(cfg: DictConfig, data_config: SeriesDataConfig, fold_index: int=0
                  , transform: NiftiTransform|None=None, is_train: bool=False
                  ) -> NiftiDataset:
    data_indices = data_config.train_data_indices[fold_index] if is_train \
        else data_config.valid_data_indices[fold_index]
    
    return hydra.utils.instantiate(
        cfg.data.nifti_train_dataset
        , df_indices=data_indices
        , label_df=data_config.data_label_df
        , transform=transform
        )

def build_transform(data_transform_cfg: DictConfig) -> NiftiTransform:
    return hydra.utils.instantiate(data_transform_cfg)

def build_raw_data(cfg: DictConfig):
    data_root_path = cfg.paths.data_root_dir
    label_path = os.path.join(data_root_path, cfg.data.label_file_name)
    series_root_path = os.path.join(data_root_path, cfg.data.series_data_folder_name)
    raw_data_root_path = os.path.join(data_root_path, cfg.data.preprocess_data_folder_name)

    label_df = pd.read_csv(label_path)

    if os.path.exists(raw_data_root_path):
        if len(glob.glob(raw_data_root_path + '*.npz')) >= len(label_df):
            return

    os.makedirs(raw_data_root_path, exist_ok=True)

    preprocessor = DICOMPreprocessor((cfg.data.input_channels, cfg.data.img_size, cfg.data.img_size))

    for paht_index, path_value in enumerate(os.walk(series_root_path)):
        root, folders, files = path_value

        for folder_index, folder in enumerate(tqdm(folders)):
            series_uid, modality, image_arrays = preprocessor(os.path.join(root, folder))

            np.savez(os.path.join(raw_data_root_path, series_uid + '.npz'), image_arrays)
            
            # if 50 <= folder_index:
            #     break
        
        if 0 < len(folders):
            break

def test_raw_data(args: argparse.Namespace):
    data_root_path = args.data_path
    if not data_root_path.endswith(os.path.sep) and not data_root_path.endswith('/'):
        # Ensure output dir is valid for later use
        data_root_path += os.path.sep
    
    raw_data_root_path = os.path.join(data_root_path, args.preprocess_data_folder_name)
    
    if not os.path.exists(raw_data_root_path):
        return
    
    for paht_index, path_value in enumerate(os.walk(raw_data_root_path)):
        root, folders, files = path_value
        
        for file_index, file in enumerate(tqdm(files)):
            if not file.endswith('.npz'):
                continue
            
            file_stream = np.load(os.path.join(raw_data_root_path, file))
            image = np.transpose(file_stream['arr_0'], (1, 2, 0))
            
            # plt.imshow(image[:, :, image.shape[2] // 2])
            # plt.show()
