
import numpy as np
import pandas as pd

import torch


class VesselROIBBoxDataset(torch.utils.data.Dataset):
    def __init__(self
                 , num_series_slide: int
                 , depth_size: int
                 , image_size: list[int]
                 ):
        self.num_series_slide = num_series_slide
        self.depth_size = depth_size
        self.image_size = list(image_size)
    
    def initialize(self
                 , meta_data: pd.DataFrame
                 ):
        self.meta_data = meta_data

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self.meta_data.iloc[index]

        volume = np.load(row.window_file).astype(np.int32)

        if volume.shape[0] < self.num_series_slide:
            volume = volume[np.linspace(0, len(volume) - 1, self.num_series_slide).astype(np.int16)]
        
        volume = volume.reshape(volume.shape[0] // self.depth_size, self.depth_size
                                , *volume.shape[-2:])

        volume = torch.as_tensor((volume - volume.min()) / (volume.max() - volume.min())).half()

        volume = torch.nn.functional.interpolate(volume, self.image_size, mode='bilinear')

        label = torch.as_tensor([row.x1, row.x2, row.y1, row.y2])

        return volume, label, str(row.window_file)
    
    def __len__(self) -> int:
        return len(self.meta_data)
