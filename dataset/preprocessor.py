
import os
import numpy as np

import pydicom
import cv2
from scipy import ndimage


class DICOMPreprocessor:
    def __init__(self, target_shape: tuple[int, int, int] = (32, 352, 352)):
        self.target_depth, self.target_height, self.target_width = target_shape

    def __call__(self, series_path: str) -> tuple[str, str, np.ndarray]:
        series_uid = os.path.basename(series_path)
        dicom_datasets = []

        for paht_index, path_value in enumerate(os.walk(series_path)):
            root, folders, files = path_value
            for file_index, file in enumerate(files):
                if file.endswith('.dcm'):
                    dataset = pydicom.dcmread(os.path.join(root, file), force=True)
                    dicom_datasets.append(dataset)
        
        if 0 >= len(dicom_datasets):
            print(f'Invalid dicom series path, series_path: {series_path}')
            return series_uid, 'None', np.ndarray([])
        
        first_image = dicom_datasets[0]
        modality = getattr(first_image, 'Modality')
        if None == modality:
            print(f'invalid modality, series_uid: {series_uid}, file: {first_image}')
        
        if len(dicom_datasets) == 1 and first_image.pixel_array.ndim == 3:
            slope = getattr(first_image, 'RescaleSlope', 1)
            intercept = getattr(first_image, 'RescaleIntercept', 0)
            sorted_slices = [(slope, intercept, first_image.pixel_array[depth_index]) for depth_index in range(first_image.pixel_array.shape[0])]
        else:
            slice_info = self.extract_slice_info(dicom_datasets)
            sorted_slices = []
            for sorted_slice in sorted(slice_info, key=lambda x: x['z_position']):
                ds = sorted_slice['dataset']
                slope = getattr(ds, 'RescaleSlope', 1)
                intercept = getattr(ds, 'RescaleIntercept', 0)
                sorted_slices.append((slope, intercept, ds.pixel_array))

        image_arrays = []

        for slice_data in sorted_slices:
            slope, intercept, pixel_array = slice_data
            
            # Get pixel array
            image = pixel_array.astype(np.float32)
            
            # Apply RescaleSlope and RescaleIntercept
            if slope != 1 or intercept != 0:
                image = image * float(slope) + float(intercept)

            normalized_image = self.apply_windowing_or_normalize(modality, image)
            resized_img = cv2.resize(normalized_image, (self.target_width, self.target_height))            
            image_arrays.append(resized_img)
        
        image_arrays = np.stack(image_arrays, axis=0)

        image_arrays = self.resize_volume_3d(image_arrays)
        
        return series_uid, modality, image_arrays

    def extract_slice_info(self, datasets: list[pydicom.Dataset]) -> list[dict]:
        """
        Extract position information for each slice
        """
        slice_info = []
        
        for i, ds in enumerate(datasets):
            info = {
                'dataset': ds,
                'index': i,
                'instance_number': getattr(ds, 'InstanceNumber', i),
            }
            
            # Get z-coordinate from ImagePositionPatient
            try:
                position = getattr(ds, 'ImagePositionPatient', None)
                if position is not None and len(position) >= 3:
                    info['z_position'] = float(position[2])
                else:
                    # Fallback: use InstanceNumber
                    info['z_position'] = float(info['instance_number'])
                    #print("ImagePositionPatient not found, using InstanceNumber")
            except Exception as e:
                info['z_position'] = float(i)
                #print(f"Failed to extract position info: {e}")
            
            slice_info.append(info)
        
        return slice_info

    def apply_windowing_or_normalize(self, modality: str, img: np.ndarray) -> np.ndarray:
        """
        Apply windowing or statistical normalization
        """
        if modality == 'CT':
            # # Windowing processing (for CT/CTA)
            # img_min = center - width / 2
            # img_max = center + width / 2
            
            # windowed = np.clip(img, img_min, img_max)
            # windowed = (windowed - img_min) / (img_max - img_min + 1e-7)
            # result = (windowed * 255).astype(np.uint8)
            
            # #print(f"Applied windowing: [{img_min:.1f}, {img_max:.1f}] → [0, 255]")
            # return result
            
            # Statistical normalization (for CT as well)
            # Normalize using 1-99 percentiles
            p1, p99 = np.percentile(img, [1, 99])
            p1, p99 = 0, 500
            
            if p99 > p1:
                normalized = np.clip(img, p1, p99)
                normalized = (normalized - p1) / (p99 - p1)
                result = (normalized * 255).astype(np.uint8)
                
                #print(f"Applied statistical normalization: [{p1:.1f}, {p99:.1f}] → [0, 255]")
                return result
            else:
                # Fallback: min-max normalization
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    normalized = (img - img_min) / (img_max - img_min)
                    result = (normalized * 255).astype(np.uint8)
                    #print(f"Applied min-max normalization: [{img_min:.1f}, {img_max:.1f}] → [0, 255]")
                    return result
                else:
                    # If image has no variation
                    #print("Image has no variation, returning zeros")
                    return np.zeros_like(img, dtype=np.uint8)
        
        else:
            # Statistical normalization (for MR)
            # Normalize using 1-99 percentiles
            p1, p99 = np.percentile(img, [1, 99])
            
            if p99 > p1:
                normalized = np.clip(img, p1, p99)
                normalized = (normalized - p1) / (p99 - p1)
                result = (normalized * 255).astype(np.uint8)
                
                #print(f"Applied statistical normalization: [{p1:.1f}, {p99:.1f}] → [0, 255]")
                return result
            else:
                # Fallback: min-max normalization
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    normalized = (img - img_min) / (img_max - img_min)
                    result = (normalized * 255).astype(np.uint8)
                    #print(f"Applied min-max normalization: [{img_min:.1f}, {img_max:.1f}] → [0, 255]")
                    return result
                else:
                    # If image has no variation
                    #print("Image has no variation, returning zeros")
                    return np.zeros_like(img, dtype=np.uint8)

    def resize_volume_3d(self, volume: np.ndarray) -> np.ndarray:
        """
        Resize 3D volume to target size
        """
        current_shape = volume.shape
        target_shape = (self.target_depth, self.target_height, self.target_width)
        
        if current_shape == target_shape:
            return volume
        
        #print(f"Resizing volume from {current_shape} to {target_shape}")
        
        # 3D resizing using scipy.ndimage
        zoom_factors = [
            target_shape[i] / current_shape[i] for i in range(3)
        ]
        
        # Resize with linear interpolation
        resized_volume = ndimage.zoom(volume, zoom_factors, order=1, mode='nearest')
        
        # Clip to exact size just in case
        resized_volume = resized_volume[:self.target_depth, :self.target_height, :self.target_width]
        
        # Padding if necessary
        pad_width = [
            (0, max(0, self.target_depth - resized_volume.shape[0])),
            (0, max(0, self.target_height - resized_volume.shape[1])),
            (0, max(0, self.target_width - resized_volume.shape[2]))
        ]
        
        if any(pw[1] > 0 for pw in pad_width):
            resized_volume = np.pad(resized_volume, pad_width, mode='edge')
        
        #print(f"Final volume shape: {resized_volume.shape}")
        return resized_volume.astype(np.uint8)
