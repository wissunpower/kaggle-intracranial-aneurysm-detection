
# https://www.kaggle.com/datasets/harshitsheoran/rsna2025-training-code
# ./try5_seg/create_data1.ipynb

import os
from collections import Counter

import rootutils

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import hydra
import glob
from tqdm.auto import tqdm
import pydicom

import torch

rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

EVAL_RESOLVER_NAME = "eval"
if not OmegaConf.has_resolver(EVAL_RESOLVER_NAME):
    OmegaConf.register_new_resolver(EVAL_RESOLVER_NAME, eval)

LEN_RESOLVER_NAME = "len"
if not OmegaConf.has_resolver(LEN_RESOLVER_NAME):
    OmegaConf.register_new_resolver(LEN_RESOLVER_NAME, len)

from utils.log import logger


class DICOMToNPYPreprocessor:
    def __init__(self
                 , crop: DictConfig
                 , resize: DictConfig
                 ):
        self.crop_opt = crop
        self.resize_opt = resize

        if self.crop_opt.enable:
            self.crop_info = np.load(crop.info_file_path, allow_pickle=True).item()

    def __call__(self, dcm_set_path: str) -> tuple[np.ndarray, list]:
        files = np.array(glob.glob(os.path.join(dcm_set_path, '*.dcm')))

        dcms = [pydicom.dcmread(file) for file in files]

        shapes = [(d.Rows, d.Columns) for d in dcms]
        most_common_shape = Counter(shapes).most_common(1)[0][0]
        
        valid_data = []
        for d, file in zip(dcms, files):
            if (d.Rows, d.Columns) == most_common_shape:
                valid_data.append([d, file])
        
        if len(valid_data) > 1:
            valid_data.sort(key=lambda x: float(x[0].ImagePositionPatient[2]))
        
        #volume = np.stack([dcm.pixel_array for dcm in valid_dcms])
        
        #t = time.time()
        volume = np.stack([dcm[0].pixel_array for dcm in valid_data])
        
        if volume.shape[0] == 1:
            volume = volume[0]
            valid_data = [valid_data[0]] * volume.shape[0]
        
        if self.crop_opt.enable:
            series_uid = os.path.basename(dcm_set_path)
            height, width = volume.shape[-2:]
            x1, x2, y1, y2 = self.crop_info[series_uid]
            volume = volume[:
                      , int(y1 * height * 0.9):int(y2 * height * 1.1)
                      , int(x1 * width * 0.9):int(x2 * width * 1.1)]

        if self.resize_opt.enable:
            height, width = self.resize_opt.target_shape
            if volume.shape[-2] > height or volume.shape[-1] > width:
                volume = torch.from_numpy(volume.astype(np.float32)).unsqueeze(0)
                volume = torch.nn.functional.interpolate(volume, (height, width)
                                                         , mode=self.resize_opt.mode, align_corners=False)
                volume = volume.squeeze(0).numpy()
        
        return volume, valid_data


def preprocess(cfg: DictConfig):
    logger.info("Setting Configuration.. : ")
    logger.info(cfg)
    print("----------------------------------------------------------")

    preprocessor: DICOMToNPYPreprocessor = hydra.utils.instantiate(cfg.preprocess)
    
    os.makedirs(cfg.output_path, exist_ok=True)

    for root, dirs, files in os.walk(cfg.series_data_path):
        if 0 >= len(dirs):
            continue

        for sub_dir in tqdm(dirs):
            volume, valid_data = preprocessor(os.path.join(root, sub_dir))

            for i, (slc, slc_data) in enumerate(zip(volume, valid_data)):
                pos = int(i + 1)
                np.save(os.path.join(cfg.output_path, f'{sub_dir}_I_{pos}.npy'), slc)


@hydra.main(version_base="1.3", config_path="../../_configs", config_name="dcm_to_npy.yaml")
def main(cfg: DictConfig):
    preprocess(cfg)


if __name__ == '__main__':
    main()
