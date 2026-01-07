
# https://www.kaggle.com/datasets/harshitsheoran/rsna2025-training-code
# ./try5_seg/create_data1.ipynb

import os

import numpy as np
import pandas as pd
import glob
from tqdm.auto import tqdm
import nibabel as nib
from joblib import Parallel, delayed


class VesselSegToROIBBoxPreprocessor:
    def __init__(self
                 , data_root_path: str
                 , segment_folder_name: str
                 , seg_output_name: str
                 , total_output_name: str
                 , series_slide_fileset_path: str
                 , series_slide_metainfo_file_name: str
                 , num_series_slide: int
                 ):
        self.data_root_path = data_root_path
        self.segment_folder_name = segment_folder_name
        self.seg_output_name = seg_output_name
        self.total_output_name = total_output_name
        self.series_slide_fileset_path = series_slide_fileset_path
        self.num_series_slide = num_series_slide
        self.data_sampled = pd.read_csv(data_root_path + series_slide_metainfo_file_name)

    def __call__(self):
        seg_output_path = self.data_root_path + self.seg_output_name
        os.makedirs(seg_output_path, exist_ok=True)

        files = np.array(glob.glob(self.data_root_path + self.segment_folder_name + '/*_cowseg.nii'))
        
        DAT = {col: [] for col in ['SeriesUID', 'window_file', 'x1', 'x2', 'y1', 'y2', 'z1', 'z2']}
        
        for file in tqdm(files):
            vol, vol_seg = nib.load(file.replace('_cowseg', '')), nib.load(file)
            volume = vol.get_fdata().astype(np.int16)
            volume_seg = vol_seg.get_fdata().astype(np.uint8)
            
            volume = volume.T[:, ::-1]
            volume_seg = volume_seg.T[:, ::-1]
            
            iid = file.split('/')[-1].split('\\')[-1].split('_')[0]
            
            rows = self.data_sampled[self.data_sampled.SeriesUID==iid]
            
            if not len(rows):
                continue
            
            rows = rows.iloc[np.linspace(0, len(rows)-1, self.num_series_slide).astype(int)]
            
            volume = volume[np.linspace(0, len(volume)-1, self.num_series_slide).astype(int)]
            
            volume_orig = np.stack([np.load(self.series_slide_fileset_path + f"{iid}_I_{i}.npy") for i in rows.InstanceNumber.values])
            
            coords = np.argwhere(volume_seg>0)
            z1, y1, x1 = coords.min(0) / volume_seg.shape
            z2, y2, x2 = coords.max(0) / volume_seg.shape
            
            window_file1 = f"{seg_output_path}/{iid}_segsampled.npy"
            np.save(window_file1, volume)
            window_file2 = f"{seg_output_path}/{iid}_origsampled.npy"
            np.save(window_file2, volume_orig)
            
            for win_file in [window_file1, window_file2]:
                DAT['SeriesUID'].append(iid)
                DAT['window_file'].append(win_file)
                DAT['x1'].append(x1)
                DAT['x2'].append(x2)
                DAT['y1'].append(y1)
                DAT['y2'].append(y2)
                DAT['z1'].append(z1)
                DAT['z2'].append(z2)
            
            #break
        
        data = pd.DataFrame(DAT)
        data.to_csv(self.data_root_path + self.seg_output_name + '.csv', index=False)

        total_output_path = self.data_root_path + self.total_output_name
        os.makedirs(total_output_path, exist_ok=True)
        
        NUM_THREADS = 4
        
        def process_series_group(group_tuple, output_folder, num_samples):
            series_uid, group_df = group_tuple
            
            rows = group_df.iloc[np.linspace(0, len(group_df) - 1, num_samples).astype(int)]
            
            volume_orig = np.stack([np.load(self.series_slide_fileset_path + f"{series_uid}_I_{i}.npy") for i in rows.InstanceNumber.values])
            
            output_file = f"{output_folder}/{series_uid}_origsampled.npy"
            np.save(output_file, volume_orig)
        
        grouped_data = self.data_sampled.groupby('SeriesUID')
        
        print(f"Processing {grouped_data.ngroups} series with {NUM_THREADS} threads...")
        
        Parallel(n_jobs=NUM_THREADS)(
            delayed(process_series_group)(group, total_output_path, self.num_series_slide) 
            for group in tqdm(grouped_data, total=grouped_data.ngroups)
        )
        
        print("Processing complete.")
