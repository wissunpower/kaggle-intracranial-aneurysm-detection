
import os
import numpy as np

import torch
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIOWithReorient

from dataset import NiftiTransform


class VesselSegmentDataset(torch.utils.data.Dataset):
    def __init__(self
                 , label_class_names
                 , volumes_path: str
                 , labels_path: str
                 ):
        '''
            self.num_classes is 'Aneurysm Present' and 13 types of vessel parts = 14
            self.num_mask_classes is back-ground and 13 types of vessel parts(fore-ground) = 14
        '''
        self.label_class_names = label_class_names
        self.num_classes = len(self.label_class_names)
        # except 'Aneurysm Present' only vessel part
        self.vessel_classes = self.num_classes - 1
        # include back-ground
        self.num_mask_classes = 1 + self.vessel_classes

        self.volumes_path = volumes_path
        self.labels_path = labels_path
        self.transform: NiftiTransform|None = None

        self.nifti_io_helper = SimpleITKIOWithReorient()

        self.uids = list[str]()
    
    def initialize(self
                 , uids: list[str]
                 , transform: NiftiTransform|None=None
                 ):
        self.uids = uids
        self.transform = transform

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        uid = self.uids[index % self.__len__()]
        volume, seg_label = self.get_raw_data(uid)

        if self.transform is not None:
            volume = self.transform(volume)
            seg_label = self.transform(seg_label)
        
        volume = torch.from_numpy(volume).float()

        seg_onehot = torch.zeros((self.num_mask_classes, *seg_label.shape[1:]))
        seg_onehot.scatter_(0, torch.from_numpy(seg_label.astype(int)), 1)

        return volume, seg_onehot
    
    def get_raw_data(self, uid: str) -> tuple[np.ndarray, np.ndarray]:
        volume_file_path = os.path.join(self.volumes_path, uid + '_0000.nii.gz')
        volume, _ = self.nifti_io_helper.read_images([volume_file_path], orientation="RAS")
        
        label_file_path = os.path.join(self.labels_path, uid + '.nii.gz')
        seg_label, _ = self.nifti_io_helper.read_images([label_file_path], orientation="RAS")

        return volume, seg_label
    
    def __len__(self) -> int:
        return len(self.uids)
