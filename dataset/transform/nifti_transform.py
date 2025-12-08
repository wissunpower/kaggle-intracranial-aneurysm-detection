
import numpy as np

from omegaconf import DictConfig


class NiftiTransform:
    def __init__(self, augmentations: DictConfig):
        self.augmentations = augmentations

    def __call__(self, volume: np.ndarray) -> np.ndarray:
        for _, augmentation in self.augmentations.items():
            volume = augmentation(volume)
        
        return volume
