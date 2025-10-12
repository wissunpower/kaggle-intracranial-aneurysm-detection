
import sys, os
import numpy as np
import pandas as pd

from tqdm.auto import tqdm

import pydicom
import SimpleITK as sitk
import matplotlib.pyplot as plt
import ipywidgets as widgets


SERIES_ROOT_PATH = 'F:/ml_data_resource/kaggle/intracranial_aneurysm_detection/series/'
LABEL_PATH = 'F:/ml_data_resource/kaggle/intracranial_aneurysm_detection/train.csv'


def load_dicom_series(dcm_folder_path: str) -> tuple[sitk.Image, dict[str, str]]:
    """
    DICOM 폴더 내의 모든 DICOM 파일을 3D 이미지로 읽기
    """
    reader = sitk.ImageSeriesReader()  # SimpleITK의 ImageSeriesReader 객체를 생성
    dicom_series = reader.GetGDCMSeriesFileNames(dcm_folder_path)  # 폴더 내의 모든 DICOM 파일들의 이름 추출
    reader.SetFileNames(dicom_series)  # 읽어올 DICOM 파일들의 이름을 설정

    # Enable loading of metadata, including private tags
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()

    dicom_images = reader.Execute()  # DICOM 파일들을 읽어서 3D 이미지로 생성
    properties = dict[str, str]()

    # Access metadata from the reader or the first slice's metadata dictionary
    if reader.HasMetaDataKey(0, "0008|0060"): # Check for the first slice
        modality = reader.GetMetaData(0, "0008|0060")
        properties['Modality'] = modality
    #     print(f"Series Modality: {modality}")
    else:
        print("Modality information (0008|0060) not found in series metadata.")

    return dicom_images, properties  # 3D 이미지를 반환

def dicom_to_numpy(dicom_images: sitk.Image) -> np.ndarray:
    """
    DICOM 이미지를 3D NumPy 배열로 변환
    """
    return sitk.GetArrayFromImage(dicom_images)  # SimpleITK 이미지를 NumPy 배열로 변환하여 반환

def display_image(array):
    """
    3D NumPy 배열을 슬라이서 위젯을 사용하여 표시
    """
    # 슬라이드를 사용하여 슬라이스를 스크롤
    def view_image(slice_index):
        plt.figure(figsize=(10, 10))  # 이미지를 표시할 Figure를 설정
        plt.imshow(array[slice_index], cmap='gray')  # 현재 슬라이스 이미지를 회색조로 표시
        plt.title(f'Slice {slice_index}')  # 슬라이스 번호를 제목으로 설정
        plt.show()  # 이미지를 출력

    slice_slider = widgets.IntSlider(min=0, max=array.shape[0] - 1, step=1, description='Slice:')  # 슬라이더를 생성
    widgets.interact(view_image, slice_index=slice_slider)  # 슬라이더와 view_image 함수를 연결하여 상호작용


def print_stat(series_root_path: str, label_path: str) -> None:

    label_df = pd.read_csv(label_path)
    stat = dict[str, dict[str, int]]()

    for root, folders, files in os.walk(series_root_path):
        for index, folder in enumerate(tqdm(folders)):
            dicom_images, properties = load_dicom_series(os.path.join(root, folder))

            dicom_array = dicom_to_numpy(dicom_images)

            modality = properties['Modality']

            if modality in stat:
                pass
            else:
                stat[modality] = {
                    'data_count': 0,
                    'channel_max': 0, 'height_max': 0, 'width_max': 0,
                    'value_max': 0, 'value_min': sys.maxsize,
                    'Left Infraclinoid Internal Carotid Artery': 0,
                    'Right Infraclinoid Internal Carotid Artery': 0,
                    'Left Supraclinoid Internal Carotid Artery': 0,
                    'Right Supraclinoid Internal Carotid Artery': 0,
                    'Left Middle Cerebral Artery': 0,
                    'Right Middle Cerebral Artery': 0,
                    'Anterior Communicating Artery': 0,
                    'Left Anterior Cerebral Artery': 0,
                    'Right Anterior Cerebral Artery': 0,
                    'Left Posterior Communicating Artery': 0,
                    'Right Posterior Communicating Artery': 0,
                    'Basilar Tip': 0,
                    'Other Posterior Circulation': 0,
                    'Aneurysm Present': 0
                    }

            stat[modality]['data_count'] += 1
            if 3 <= len(dicom_array.shape):
                stat[modality]['channel_max'] = max(dicom_array.shape[0], stat[modality]['channel_max'])
                stat[modality]['height_max'] = max(dicom_array.shape[1], stat[modality]['height_max'])
                stat[modality]['width_max'] = max(dicom_array.shape[2], stat[modality]['width_max'])
            stat[modality]['value_max'] = max(dicom_array.max(), stat[modality]['value_max'])
            stat[modality]['value_min'] = min(dicom_array.min(), stat[modality]['value_min'])

            label_result = label_df.loc[label_df['SeriesInstanceUID'] == folder]
            stat[modality]['Left Infraclinoid Internal Carotid Artery'] += int(label_result['Left Infraclinoid Internal Carotid Artery'].iloc[0])
            stat[modality]['Right Infraclinoid Internal Carotid Artery'] += int(label_result['Right Infraclinoid Internal Carotid Artery'].iloc[0])
            stat[modality]['Left Supraclinoid Internal Carotid Artery'] += int(label_result['Left Supraclinoid Internal Carotid Artery'].iloc[0])
            stat[modality]['Right Supraclinoid Internal Carotid Artery'] += int(label_result['Right Supraclinoid Internal Carotid Artery'].iloc[0])
            stat[modality]['Left Middle Cerebral Artery'] += int(label_result['Left Middle Cerebral Artery'].iloc[0])
            stat[modality]['Right Middle Cerebral Artery'] += int(label_result['Right Middle Cerebral Artery'].iloc[0])
            stat[modality]['Anterior Communicating Artery'] += int(label_result['Anterior Communicating Artery'].iloc[0])
            stat[modality]['Left Anterior Cerebral Artery'] += int(label_result['Left Anterior Cerebral Artery'].iloc[0])
            stat[modality]['Right Anterior Cerebral Artery'] += int(label_result['Right Anterior Cerebral Artery'].iloc[0])
            stat[modality]['Left Posterior Communicating Artery'] += int(label_result['Left Posterior Communicating Artery'].iloc[0])
            stat[modality]['Right Posterior Communicating Artery'] += int(label_result['Right Posterior Communicating Artery'].iloc[0])
            stat[modality]['Basilar Tip'] += int(label_result['Basilar Tip'].iloc[0])
            stat[modality]['Other Posterior Circulation'] += int(label_result['Other Posterior Circulation'].iloc[0])
            stat[modality]['Aneurysm Present'] += int(label_result['Aneurysm Present'].iloc[0])
    
    print(stat)

def print_stat2(series_root_path: str, label_path: str) -> None:

    label_df = pd.read_csv(label_path)
    stat = dict[str, dict[str, int]]()

    for paht_index, path_value in enumerate(tqdm(os.walk(series_root_path))):
        root, folders, files = path_value
        series_uid = os.path.basename(root)
        modality = 'None'
        image_arrays = []
        for file_index, file in enumerate(files):
            if file.endswith('.dcm'):
                dicom_file = pydicom.dcmread(os.path.join(root, file), force=True)

                modality = getattr(dicom_file, 'Modality')                
                if None == modality:
                    print(f'invalid modality, series_uid: {series_uid}, file: {file}')

                # Get pixel data
                image_array = dicom_file.pixel_array.astype(np.float32)
                
                # For 3D volume case (multiple frames) - select middle frame
                if 3 <= image_array.ndim:
                    print(f'invalid dim, series_uid: {series_uid}, file: {file}')
                
                image_arrays.append(image_array)
        
        if 0 >= len(image_arrays):
            continue

        image_arrays = np.stack(image_arrays, axis=0)
        
        if modality in stat:
            pass
        else:
            stat[modality] = {
                'data_count': 0,
                'channel_max': 0, 'height_max': 0, 'width_max': 0,
                'value_max': 0, 'value_min': sys.maxsize,
                'Left Infraclinoid Internal Carotid Artery': 0,
                'Right Infraclinoid Internal Carotid Artery': 0,
                'Left Supraclinoid Internal Carotid Artery': 0,
                'Right Supraclinoid Internal Carotid Artery': 0,
                'Left Middle Cerebral Artery': 0,
                'Right Middle Cerebral Artery': 0,
                'Anterior Communicating Artery': 0,
                'Left Anterior Cerebral Artery': 0,
                'Right Anterior Cerebral Artery': 0,
                'Left Posterior Communicating Artery': 0,
                'Right Posterior Communicating Artery': 0,
                'Basilar Tip': 0,
                'Other Posterior Circulation': 0,
                'Aneurysm Present': 0
                }
        
        stat[modality]['data_count'] += 1
        if 3 <= len(image_arrays.shape):
            stat[modality]['channel_min'] = min(image_arrays.shape[0], stat[modality]['channel_min'])
            stat[modality]['channel_max'] = max(image_arrays.shape[0], stat[modality]['channel_max'])
            stat[modality]['height_min'] = min(image_arrays.shape[1], stat[modality]['height_min'])
            stat[modality]['height_max'] = max(image_arrays.shape[1], stat[modality]['height_max'])
            stat[modality]['width_min'] = min(image_arrays.shape[2], stat[modality]['width_min'])
            stat[modality]['width_max'] = max(image_arrays.shape[2], stat[modality]['width_max'])
        stat[modality]['value_max'] = max(image_arrays.max(), stat[modality]['value_max'])
        stat[modality]['value_min'] = min(image_arrays.min(), stat[modality]['value_min'])
        
        label_result = label_df.loc[label_df['SeriesInstanceUID'] == series_uid]
        stat[modality]['Left Infraclinoid Internal Carotid Artery'] += int(label_result['Left Infraclinoid Internal Carotid Artery'].iloc[0])
        stat[modality]['Right Infraclinoid Internal Carotid Artery'] += int(label_result['Right Infraclinoid Internal Carotid Artery'].iloc[0])
        stat[modality]['Left Supraclinoid Internal Carotid Artery'] += int(label_result['Left Supraclinoid Internal Carotid Artery'].iloc[0])
        stat[modality]['Right Supraclinoid Internal Carotid Artery'] += int(label_result['Right Supraclinoid Internal Carotid Artery'].iloc[0])
        stat[modality]['Left Middle Cerebral Artery'] += int(label_result['Left Middle Cerebral Artery'].iloc[0])
        stat[modality]['Right Middle Cerebral Artery'] += int(label_result['Right Middle Cerebral Artery'].iloc[0])
        stat[modality]['Anterior Communicating Artery'] += int(label_result['Anterior Communicating Artery'].iloc[0])
        stat[modality]['Left Anterior Cerebral Artery'] += int(label_result['Left Anterior Cerebral Artery'].iloc[0])
        stat[modality]['Right Anterior Cerebral Artery'] += int(label_result['Right Anterior Cerebral Artery'].iloc[0])
        stat[modality]['Left Posterior Communicating Artery'] += int(label_result['Left Posterior Communicating Artery'].iloc[0])
        stat[modality]['Right Posterior Communicating Artery'] += int(label_result['Right Posterior Communicating Artery'].iloc[0])
        stat[modality]['Basilar Tip'] += int(label_result['Basilar Tip'].iloc[0])
        stat[modality]['Other Posterior Circulation'] += int(label_result['Other Posterior Circulation'].iloc[0])
        stat[modality]['Aneurysm Present'] += int(label_result['Aneurysm Present'].iloc[0])
    
    print(stat)


if __name__ == '__main__':
    # # DICOM 폴더 내의 모든 DICOM 파일을 3D 이미지로 읽기
    # dicom_images = load_dicom_series(SERIES_ROOT_PATH + '1.2.826.0.1.3680043.8.498.10102361048562788202568222767625052953/')
    
    # # DICOM 이미지를 3D NumPy 배열로 변환
    # dicom_array = dicom_to_numpy(dicom_images)
    
    # # 3D NumPy 배열을 표시
    # display_image(dicom_array)

    print_stat2(SERIES_ROOT_PATH, LABEL_PATH)
