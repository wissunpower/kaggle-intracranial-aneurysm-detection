
import sys, random
import numpy as np

import albumentations as albu


class SeriesTransform:
    def __init__(self, img_size: int, img_crop_size: int):
        self.img_size = img_size
        self.img_crop_size = img_crop_size
        self.step_count = 0
        
        self.init_aug_hyper_params()
        self.init_crop()
        self.init_albu()

    def init_aug_hyper_params(self):
        self.RANDOM_CROP_PROB = 1.0

        self.CUT_OUT_PROB = 0.9375
        self.CUT_OUT_AREA_RATE = 0.3

    def init_crop(self):
        crop_min_size = min(self.img_size, self.img_crop_size)
        crop_max_size = max(self.img_size, self.img_crop_size)

        self.crop_strides = [int(stride) for stride in range(crop_min_size, crop_max_size + 1, 32)]
        # self.crop_strides = [self.img_crop_size]

        self.update_crop_setting()
    
    def init_albu(self):
        ROTATE_PROB = 0.625
        ROTATE_DEGREE = (-12, 12)
        self.albu_affine_rotater = albu.Rotate(ROTATE_DEGREE, p=ROTATE_PROB)

        LIGHT_PROB = 0.375
        self.albu_light_set = [
            albu.RandomBrightnessContrast(p=LIGHT_PROB),
            albu.RandomGamma(p=LIGHT_PROB)
        ]

        NOISE_PROB = 0.125
        self.albu_noise_set = [
            albu.GaussNoise(p=NOISE_PROB),
            albu.ShotNoise(p=NOISE_PROB),
            albu.AdditiveNoise(p=NOISE_PROB),
            albu.MultiplicativeNoise(p=NOISE_PROB)
            ]
        
        BLUR_PROB = 0.125
        self.albu_blur_set = [
            albu.Blur(p=BLUR_PROB),
            albu.GaussianBlur(p=BLUR_PROB),
            albu.MedianBlur(p=BLUR_PROB),
            albu.MotionBlur(p=BLUR_PROB),
            albu.ZoomBlur(p=BLUR_PROB),
            albu.GlassBlur(p=BLUR_PROB),
            albu.AdvancedBlur(p=BLUR_PROB)
            ]
        
        EQUALIZE_PROB = 0.25
        self.albu_equalize_set = [
            albu.Equalize(p=EQUALIZE_PROB),
            albu.CLAHE(p=EQUALIZE_PROB),
            ]
        
    def __call__(self, images: np.ndarray) -> np.ndarray:
        # albumentations augmentation
        albu_transform = self.get_rand_albu_transform()
        images = albu_transform(images=images)['images']

        # cut out category
        if random.random() < self.CUT_OUT_PROB:
            images = self.cut_out_augment(images)

        return images
    
    def cut_out_augment(self, image: np.ndarray) -> np.ndarray:
        depth, height, width = image.shape
        area = float(height) * float(width)
        min_size = max(int(round(np.sqrt(area * 0.05))), 0)
        max_size = int(round(np.sqrt(area * 0.5)))

        blocked_area = area * self.CUT_OUT_AREA_RATE

        size_alpha = np.random.uniform(low=min_size, high=min(max_size, width))
        size_alpha = max(size_alpha, sys.float_info.epsilon)
        size_beta = blocked_area / size_alpha
        
        cut_width, cut_height = (int(size_alpha), int(size_beta)) if random.random() < 0.5 \
                                else (int(size_beta), int(size_alpha))
        cut_width = min(cut_width, width)
        cut_height = min(cut_height, height)
        left = int(np.random.uniform(low=0, high=(width-cut_width)))
        top = int(np.random.uniform(low=0, high=(height-cut_height)))

        cut_area = np.zeros((depth, cut_height, cut_width), dtype=np.uint8)
        
        image[:, top:top+cut_height, left:left+cut_width] = cut_area

        return image

    def get_rand_albu_transform(self) -> albu.Compose:
        light = np.random.choice(self.albu_light_set)
        noiser = np.random.choice(self.albu_noise_set)
        blur = np.random.choice(self.albu_blur_set)
        equalizer = np.random.choice(self.albu_equalize_set)
        
        # return albu.Compose([
        #     self.albu_affine_rotater,
        #     self.albu_random_crop,
        #     light,
        #     noiser,
        #     blur,
        #     equalizer,
        #     ])
        
        return albu.Compose([
            self.albu_affine_rotater,
            self.albu_random_crop,
            *self.albu_light_set,
            *self.albu_noise_set,
            *self.albu_blur_set,
            *self.albu_equalize_set,
            ])

    def update_crop_setting(self):
        current_crop_size = np.random.choice(self.crop_strides)
        
        current_prob = 1.0 if random.random() < self.RANDOM_CROP_PROB else 0.0
        self.albu_random_crop \
            = albu.RandomCrop(current_crop_size, current_crop_size, p=current_prob)
    
    def decay_aug_hyper_params(self):
        DECAY_RATE = 0.8

        # self.RANDOM_CROP_PROB *= DECAY_RATE

        self.CUT_OUT_PROB *= DECAY_RATE
        # self.CUT_OUT_AREA_RATE *= DECAY_RATE
        # self.CUT_OUT_AREA_RATE = max(self.CUT_OUT_AREA_RATE, sys.float_info.epsilon)

    def step_by_batch(self):
        self.step_count += 1

        # if 0 == self.step_count % 900:
        #     self.decay_aug_hyper_params()

        self.update_crop_setting()
