
import numpy as np

from omegaconf import DictConfig

import albumentations as albu


class ImageMixedTransform:
    def __init__(self, augmentations: DictConfig):
        self.augmentations = augmentations

        albu_image_only_augmentations = []

        for _, augmentation in augmentations.items():
            if isinstance(augmentation, albu.ImageOnlyTransform):
                albu_image_only_augmentations.append(augmentation)

        self.albu_image_only_augmentations = albu.Compose(albu_image_only_augmentations)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if len(self.albu_image_only_augmentations) == len(self.augmentations):
            ''' only albu image transform exist case '''
            image = self.albu_image_only_augmentations(images=image)['images']
        else:
            for _, augmentation in self.augmentations.items():
                if isinstance(augmentation, albu.ImageOnlyTransform):
                    image = augmentation(images=image)['images']
                else:
                    image = augmentation(image)
        
        return image
