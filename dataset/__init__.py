
import sys, os
import argparse
import numpy as np
import pandas as pd
import glob

from tqdm.auto import tqdm

import torch

from dataset.config import SeriesDataConfig
from dataset.dataset import DICOMDataset
from dataset.transform import SeriesTransform
from dataset.preprocessor import DICOMPreprocessor


def build_dataset(args: argparse.Namespace, data_config: SeriesDataConfig, fold_index: int=0
                  , transform: object=None, is_train: bool=False) -> DICOMDataset:
    data_indices = data_config.train_data_indices[fold_index] if is_train \
        else data_config.valid_data_indices[fold_index]
    return DICOMDataset(args, data_config, data_indices, transform=transform)

def build_transform(args: argparse.Namespace) -> object:
    return SeriesTransform(args)

def build_raw_data(args: argparse.Namespace):
    data_root_path = args.data_path
    label_path = os.path.join(data_root_path, args.label_file_name)
    series_root_path = os.path.join(data_root_path, 'series/')
    raw_data_root_path = os.path.join(data_root_path, args.preprocess_data_folder_name)

    label_df = pd.read_csv(label_path)

    if os.path.exists(raw_data_root_path):
        if len(glob.glob(raw_data_root_path + '*.npz')) >= len(label_df):
            return

    os.makedirs(raw_data_root_path, exist_ok=True)

    preprocessor = DICOMPreprocessor((args.input_channels, args.img_size, args.img_size))

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
