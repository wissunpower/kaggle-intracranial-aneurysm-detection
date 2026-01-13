
import numpy as np

import torch


class ImageResize:
    def __init__(self
                 , target_shape: tuple[int, int]
                 , forced_scale: bool=False
                 , mode: str='bilinear'):
        self.target_shape = target_shape
        self.forced_scale = forced_scale
        self.mode = mode

    def __call__(self, image: np.ndarray) -> np.ndarray:
        target_height, target_width = self.target_shape[-2:]

        if 2 == len(image.shape):
            image = image[np.newaxis, :]
        
        rs_height, rs_width = self.target_shape[-2:] if self.forced_scale \
                                                    else self.resize_shape(image.shape[-2:])

        image = torch.from_numpy(image).unsqueeze(0)
        image = torch.nn.functional.interpolate(image, (rs_height, rs_width)
                                                 , mode=self.mode, align_corners=False)

        pad_height = max(0, target_height - rs_height)
        pad_width = max(0, target_width - rs_width)

        if 0 < pad_height or 0 < pad_width:
            image = torch.nn.functional.pad(image
                                             , (
                                                 pad_width // 2, pad_width - (pad_width // 2)
                                                 , pad_height // 2, pad_height - (pad_height // 2),)
                                             , mode='constant', value=0)

        return image.squeeze(0).numpy()
    
    def resize_shape(self
                     , input_shape: tuple[int, int]
                     ) -> tuple[int, int]:
        origin_height, origin_width = input_shape
        target_height, target_width = self.target_shape[-2:]

        image_scale = min(target_height / origin_height
                          , target_width / origin_width)
        
        return (max(1, int(round(origin_height * image_scale)))
                , max(1, int(round(origin_width * image_scale))))


class VolumeResize:
    def __init__(self
                 , target_shape: tuple[int, int, int]
                 , forced_scale: bool=False
                 , mode: str='trilinear'):
        self.target_shape = target_shape
        self.forced_scale = forced_scale
        self.mode = mode

    def __call__(self, volume: np.ndarray) -> np.ndarray:
        target_depth, target_height, target_width = self.target_shape

        if 3 == len(volume.shape):
            volume = volume[np.newaxis, :]
        
        rs_depth, rs_height, rs_width = self.target_shape if self.forced_scale \
                                                        else self.resize_shape(volume.shape[-3:])

        volume = torch.from_numpy(volume).unsqueeze(0)
        volume = torch.nn.functional.interpolate(volume, (rs_depth, rs_height, rs_width)
                                                 , mode=self.mode, align_corners=False)

        pad_depth = max(0, target_depth - rs_depth)
        pad_height = max(0, target_height - rs_height)
        pad_width = max(0, target_width - rs_width)

        if 0 < pad_depth or 0 < pad_height or 0 < pad_width:
            volume = torch.nn.functional.pad(volume
                                             , (
                                                 pad_width // 2, pad_width - (pad_width // 2)
                                                 , pad_height // 2, pad_height - (pad_height // 2)
                                                 , pad_depth // 2, pad_depth - (pad_depth // 2),)
                                             , mode='constant', value=0)

        return volume.squeeze(0).numpy()
    
    def resize_shape(self
                     , input_shape: tuple[int, int, int]
                     ) -> tuple[int, int, int]:
        origin_depth, origin_height, origin_width = input_shape
        target_depth, target_height, target_width = self.target_shape

        depth_scale = target_depth / origin_depth
        image_scale = min(target_height / origin_height
                          , target_width / origin_width)
        
        return (max(1, int(round(origin_depth * depth_scale)))
                , max(1, int(round(origin_height * image_scale)))
                , max(1, int(round(origin_width * image_scale))))
