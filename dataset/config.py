
import os, time
import numpy as np
import pandas as pd
import glob


ID_COL = 'SeriesInstanceUID'

LABEL_COLS = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present',
]

NUM_CLASSES = len(LABEL_COLS)

# All tags (other than PixelData and SeriesInstanceUID) that may be in a test set dcm file
DICOM_TAG_ALLOWLIST = [
    'BitsAllocated',
    'BitsStored',
    'Columns',
    'FrameOfReferenceUID',
    'HighBit',
    'ImageOrientationPatient',
    'ImagePositionPatient',
    'InstanceNumber',
    'Modality',
    'PatientID',
    'PhotometricInterpretation',
    'PixelRepresentation',
    'PixelSpacing',
    'PlanarConfiguration',
    'RescaleIntercept',
    'RescaleSlope',
    'RescaleType',
    'Rows',
    'SOPClassUID',
    'SOPInstanceUID',
    'SamplesPerPixel',
    'SliceThickness',
    'SpacingBetweenSlices',
    'StudyInstanceUID',
    'TransferSyntaxUID',
]


class SeriesDataConfig:
    def __init__(self
                 , data_path: str
                 , niix_path: str
                 , label_file_name: str
                 , label_weight_priority: list[str]
                 , num_fold: int):
        self.data_path = data_path
        self.niix_path = niix_path
        self.data_label_df = pd.read_csv(os.path.join(self.data_path, label_file_name))

        self.label_weight_priority = self.parse_label_weight_priority(label_weight_priority)

        self.train_data_indices, self.valid_data_indices \
            = self.split_data_index_with_fold(num_fold) if 1 < num_fold \
                else self.split_data_index()
        
        self.label_auc_weights = [1., 1., 1., 1., 1.,
                                  1., 1., 1., 1., 1.,
                                  1., 1., 1., 13.,]
        
        self.save_start_time_str = time.strftime('%Y%m%d_%H%M%S')
    
    def parse_label_weight_priority(self, label_priority: list[str]) -> list[int]:
        weight_priority = []
        
        if 0 >= len(label_priority):
            label_priority = [
                'Left Anterior Cerebral Artery',
                'Right Anterior Cerebral Artery',
                'Left Infraclinoid Internal Carotid Artery',
                'Left Posterior Communicating Artery',
                'Right Infraclinoid Internal Carotid Artery',
                'Right Posterior Communicating Artery',
                'Basilar Tip',
                'Other Posterior Circulation',
                'Left Middle Cerebral Artery',
                'Right Supraclinoid Internal Carotid Artery',
                'Right Middle Cerebral Artery',
                'Left Supraclinoid Internal Carotid Artery',
                'Anterior Communicating Artery',
                'Aneurysm Present',
            ]

        for label_name in label_priority:
            try:
                weight_priority.append(LABEL_COLS.index(label_name))
            except ValueError:
                print(f'Invalid label name: {label_name}')
        
        if 0 >= len(weight_priority):
            weight_priority.append(LABEL_COLS.index('Aneurysm Present'))
        
        return weight_priority

    def get_excepted_indices(self) -> np.ndarray:
        indices = list[int]()

        for index, row in self.data_label_df.iterrows():
            nifti_file_name_pattern = os.path.join(self.niix_path, str(row.iloc[0]), '*.nii.gz')
            nifti_files = glob.glob(nifti_file_name_pattern)
            if 0 < len(nifti_files):
                continue
            
            indices.append(index)

        return np.array(indices)

    def split_data_index_with_fold(self, num_fold: int=1) -> tuple[list[np.ndarray], list[np.ndarray]]:
        num_data = len(self.data_label_df)

        data_label_np = self.data_label_df.to_numpy()
        column_start_index = 4
        total_indices = np.arange(num_data)

        excepted_indices = self.get_excepted_indices()
        selected_mask = np.full(num_data, False)
        selected_mask[excepted_indices] = True

        indices_folds = [[] for _ in range(num_fold)]

        for label_index in self.label_weight_priority:
            positive_indices = np.where(data_label_np[:, column_start_index + label_index] != 0)[0]
            positive_indices = np.setdiff1d(positive_indices, total_indices[selected_mask])
            num_positive = len(positive_indices)

            if 0 >= num_positive:
                continue

            for fold_index in range(num_fold):
                pii = np.where(np.arange(num_positive) % num_fold == fold_index)[0]
                indices_folds[fold_index].extend(positive_indices[pii])

            selected_mask[positive_indices] = True
        
        remain_indices = total_indices[selected_mask != True]
        num_remain = len(remain_indices)

        if 0 < num_remain:
            for fold_index in range(num_fold):
                rii = np.where(np.arange(num_remain) % num_fold == fold_index)[0]
                indices_folds[fold_index].extend(remain_indices[rii])


        total_train_indices = [np.array([], np.int64) for _ in range(num_fold)]
        total_valid_indices = [np.array([], np.int64) for _ in range(num_fold)]
        
        for fold_index0 in range(num_fold):
            for fold_index1 in range(num_fold):
                if fold_index0 == fold_index1:
                    total_valid_indices[fold_index0] \
                        = np.append(total_valid_indices[fold_index0], np.array(indices_folds[fold_index1]))
                else:
                    total_train_indices[fold_index0] \
                        = np.append(total_train_indices[fold_index0], np.array(indices_folds[fold_index1]))

        for train_indices, valid_indices in zip(total_train_indices, total_valid_indices):
            np.random.shuffle(train_indices)
            np.random.shuffle(valid_indices)
            
            assert len(train_indices) == len(np.unique(train_indices)), 'Duplicate train index exists.'
            assert len(valid_indices) == len(np.unique(valid_indices)), 'Duplicate valid index exists.'
            assert len(valid_indices) + len(train_indices) == (num_data - len(excepted_indices)) \
                , 'Missing index exists.'
        
        return total_train_indices, total_valid_indices

    def split_data_index(self, valid_data_rate: float = 0.2) -> tuple[list[np.ndarray], list[np.ndarray]]:
        num_data = len(self.data_label_df)
        valid_data_rate = np.clip(valid_data_rate, 0.0, 1.0)

        data_label_np = self.data_label_df.to_numpy()
        column_start_index = 4
        total_indices = np.arange(num_data)

        excepted_indices = self.get_excepted_indices()
        selected_mask = np.full(num_data, False)
        selected_mask[excepted_indices] = True

        train_indices = []
        valid_indices = []

        for label_index in self.label_weight_priority:
            positive_indices = np.where(data_label_np[:, column_start_index + label_index] != 0)[0]
            positive_indices = np.setdiff1d(positive_indices, total_indices[selected_mask])
            num_valid = int(len(positive_indices) * valid_data_rate)

            if 0 >= num_valid:
                continue
            
            train_indices.extend(positive_indices[num_valid:])
            valid_indices.extend(positive_indices[:num_valid])

            selected_mask[positive_indices] = True
        
        remain_indices = total_indices[selected_mask != True]
        num_valid = max(int(len(remain_indices) * valid_data_rate), 1)

        train_indices.extend(remain_indices[num_valid:])
        valid_indices.extend(remain_indices[:num_valid])

        np.random.shuffle(train_indices)
        np.random.shuffle(valid_indices)

        assert len(train_indices) == len(np.unique(train_indices)), 'Duplicate train index exists.'
        assert len(valid_indices) == len(np.unique(valid_indices)), 'Duplicate valid index exists.'
        assert len(valid_indices) + len(train_indices) == (num_data - len(excepted_indices)) \
            , 'Missing index exists.'
        
        return [np.array(train_indices)], [np.array(valid_indices)]
