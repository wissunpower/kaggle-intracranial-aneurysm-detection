
import os
import shutil
import argparse
import glob

import numpy as np
import pandas as pd
import polars as pl

import torch

from utils.misc import parse_args
from dataset.config import NUM_CLASSES, SeriesDataConfig
from dataset.preprocessor import DICOMPreprocessor
from models import build_model
from evaluator import build_evaluator


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


def test(args: argparse.Namespace, data_config: SeriesDataConfig, model: torch.nn.Module
         , device: torch.device):
    evaluator = build_evaluator(args, data_config, device)
    
    _, valid_accuracy = evaluator.evaluate(model, None)
    
    print(f'Validation Accuracy: {valid_accuracy:.4f}')

# def eval():
#     args = parse_args()
#     print("Setting Arguments.. : ", args)
#     print("----------------------------------------------------------")

#     device = torch.device(torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu')
#     print(f'device: {device}')

#     data_config = SeriesDataConfig(args)

#     model = build_model(args, NUM_CLASSES)
#     model.load_state_dict(torch.load(args.model_weight_path))
#     model = model.to(device)

#     test(args, data_config, model, device)


args = parse_args()
print("Setting Arguments.. : ", args)
print("----------------------------------------------------------")

device = torch.device(torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu')
print(f'device: {device}')

data_config = SeriesDataConfig(args)

assert os.path.exists(args.model_weight_path), 'Not exist model weight path.'
assert len(glob.glob(args.model_weight_path + '*.pth')) == args.num_fold, \
    'Not match number model fold and weight file.'

model_loaders = build_model(args, NUM_CLASSES)
models = []

for paht_index, path_value in enumerate(os.walk(args.model_weight_path)):
    root, folders, files = path_value
    for file_index, file in enumerate(files):
         if file.endswith('.pth'):
            model_loaders[file_index].load_state_dict(torch.load(os.path.join(args.model_weight_path, file)))
            models.append(model_loaders[file_index].to(device))

preprocessor = DICOMPreprocessor((args.input_channels, args.img_size, args.img_size))


def predict(series_path: str) -> pl.DataFrame | pd.DataFrame:

    series_uid = os.path.basename(series_path)

    try:
        # Extract series ID
        series_uid, modality, image_arrays = preprocessor(series_path)

        image_arrays = torch.from_numpy(image_arrays).contiguous().float() / 255.
        image_arrays = image_arrays.to(device)

        predicts = []

        for index, model in enumerate(models):
            predict = model.inference(image_arrays)
            predict = predict.squeeze().cpu()

            predicts.append(predict)
        
        weights = np.ones(args.num_fold)
        weights = weights / np.sum(weights)

        predicts = np.average(predicts, axis=0, weights=weights)

        result_df = pl.DataFrame(
            data=[[series_uid] + predicts.tolist()],
            schema=[ID_COL, *LABEL_COLS],
            orient='row'
        )
    except Exception as e:
        # Return a fallback dataframe with the correct schema
        result_df = pl.DataFrame(
            data=[[series_uid] + [0.1] * len(LABEL_COLS)],
            schema=[ID_COL, *LABEL_COLS],
            orient='row'
        )
    finally:
        # This code is required to prevent "out of disk space" and "directory not empty" errors.
        # It deletes the shared folder and then immediately recreates it, ensuring it's
        # empty and ready for the next prediction.

        # shared_dir = '/kaggle/shared'
        shared_dir = './shared'
        shutil.rmtree(shared_dir, ignore_errors=True)
        os.makedirs(shared_dir, exist_ok=True)
    
    return result_df.drop(ID_COL)


if __name__ == '__main__':
    # eval()
    results = predict('F:/ml_data_resource/kaggle/intracranial_aneurysm_detection/series_debug/1.2.826.0.1.3680043.8.498.10102361048562788202568222767625052953')
    print(results.head())
