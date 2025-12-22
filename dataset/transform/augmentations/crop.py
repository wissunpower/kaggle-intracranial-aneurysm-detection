
import numpy as np

import torch


class VolumeRandomCrop:
    def __init__(self, target_shape: tuple[int, int, int]):
        self.target_shape = target_shape

    def __call__(self, volume: np.ndarray) -> np.ndarray:
        origin_depth, origin_height, origin_width = volume.shape[-3:]
        target_depth, target_height, target_width = self.target_shape

        if 3 == len(volume.shape):
            volume = volume[np.newaxis, :]

        volume = torch.from_numpy(volume).unsqueeze(0)

        pad_depth = max(0, target_depth - origin_depth)
        pad_height = max(0, target_height - origin_height)
        pad_width = max(0, target_width - origin_width)

        if 0 < pad_depth or 0 < pad_height or 0 < pad_width:
            volume = torch.nn.functional.pad(volume
                                             , (
                                                 pad_width // 2, pad_width - (pad_width // 2)
                                                 , pad_height // 2, pad_height - (pad_height // 2)
                                                 , pad_depth // 2, pad_depth - (pad_depth // 2),)
                                             , mode='constant', value=0)

        depth, height, width = volume.shape[-3:]

        start_depth = int(np.random.uniform(0, max(0, depth - target_depth)))
        start_height = int(np.random.uniform(0, max(0, height - target_height)))
        start_width = int(np.random.uniform(0, max(0, width - target_width)))

        volume = volume[:, :
                        , start_depth:start_depth+target_depth
                        , start_height:start_height+target_height
                        , start_width:start_width+target_width]

        return volume.squeeze(0).numpy()
