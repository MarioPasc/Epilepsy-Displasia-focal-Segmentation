import os
import cv2
import nibabel as nib
import numpy as np
import random
from tqdm import tqdm
import glob
from typing import Dict, List, Tuple
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
import re
from holdout import HoldOut
class DataAugmentation:

    def __init__(self, study_path: str, roi_path: str, HoldOutInstance: HoldOut) -> None:
        self.study_path = study_path
        self.roi_path = roi_path
        
        # Dictionary, Patient ID: [Slices with ROI]
        self.valid_roi_slices: Dict[str, List[int]] = {}
        # Train, validation and test sets (as lists)
        self.test_set = HoldOutInstance.test_set
        self.train_set = HoldOutInstance.train_set
        self.val_set = HoldOutInstance.val_set

        self._find_valid_roi_slices_()

    # Find valid slices for each patient. We consider a slice as valid
    # if its corresponding roi's contour has at least 5 points
    def _process_roi_file_(self, roi_file: str) -> Tuple[str, List[int]]:
        valid_slices = []
        roi_path = os.path.join(self.roi_path, roi_file)
        roi_img = nib.load(roi_path)
        roi_data = roi_img.get_fdata()

        for i in range(roi_data.shape[2]):
            slice_data = roi_data[:, :, i]
            slice_data = np.array(slice_data, dtype=np.uint8)
            contours, _ = cv2.findContours(slice_data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours and math.ceil(len(contours[0]) / 2) > 3:
                valid_slices.append(i)

        patient_id = roi_file.split('_')[0]
        return (patient_id, valid_slices)

    # Apply the function process_roi_file concurrently
    def _find_valid_roi_slices_(self) -> None:
        if not os.path.exists(self.roi_path):
            print("ROI path does not exist can't apply data augmentation.")
            return

        with ProcessPoolExecutor() as executor:
            futures = []

            for roi_file in os.listdir(self.roi_path):
                # Avoid applying data augmentation techniques to images in the test set
                if roi_file.endswith('.nii.gz'):
                    futures.append(executor.submit(self._process_roi_file_, roi_file))

            for future in as_completed(futures):
                patient_id, valid_slices = future.result()
                if valid_slices:
                    self.valid_roi_slices[patient_id] = valid_slices

    

    # Apply the selected data augmentation method.
    # The augmentation method will create two .nii.gz file, one corresponding
    # to the study img and the other to the corresponding ROI
    def apply_augmentations(self, augmentation_types: List[str], num_patients:int) -> None:
        # Avoid using test patients as input augmentation data
        test_patients = np.array([x.split("_")[0] for x in self.test_set])
        train_val_patients = np.setdiff1d(np.array(list(self.valid_roi_slices.keys())), test_patients)
        selected_patients = random.choices(train_val_patients, k=num_patients)

        with ProcessPoolExecutor() as executor:
            futures = []

            # Generate tasks for the executor for selected patients
            for patient_id in selected_patients:
                for augmentation_type in augmentation_types:
                    futures.append(executor.submit(self._apply_and_save_augmentation, patient_id, augmentation_type))

            # Use tqdm to show the progress bar
            tasks_progress = tqdm(as_completed(futures), total=len(futures), desc="Applying data augmentation")

            for future in tasks_progress:
                future.result() 

    # Finds the patient's .nii.gz files, but excludes the ones that have the augmentation type applied in its name
    def _find_files(self, path: str, patient_id: str, exclude_keywords: List[str]) -> List[str]:
        all_files = glob.glob(os.path.join(path, f'{patient_id}*.nii.gz'))
        filtered_files = [file for file in all_files if not any(keyword in file for keyword in exclude_keywords)]
        return filtered_files

    def _gamma(self, study_slice: np.ndarray, roi_slice: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        gamma_value = random.uniform(0.1, 2)
        corrected_image = np.power(study_slice / np.max(study_slice), gamma_value) * np.max(study_slice)
        return np.clip(corrected_image, a_min=0, a_max=np.max(study_slice)), roi_slice

    def _brightness(self, study_slice: np.ndarray, roi_slice: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        brightness_value = random.uniform(-15, 15)
        modified_image = study_slice + brightness_value
        return np.clip(modified_image, a_min=0, a_max=np.max(study_slice)), roi_slice

    def _shift(self, study_slice: np.ndarray, roi_slice: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        tx = random.randint(-5, 5)
        ty = random.randint(-5, 5)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        shifted_study_slice = cv2.warpAffine(study_slice, M, (study_slice.shape[1], study_slice.shape[0]),
                                             borderMode=cv2.BORDER_REFLECT)
        shifted_roi_slice = cv2.warpAffine(roi_slice, M, (roi_slice.shape[1], roi_slice.shape[0]),
                                           borderMode=cv2.BORDER_REFLECT)
        return shifted_study_slice, shifted_roi_slice

    def _flip(self, study_slice: np.ndarray, roi_slice: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Realizar un flip horizontal (de derecha a izquierda) en la imagen y en el ROI
        flipped_study_slice = cv2.flip(study_slice, 1)
        flipped_roi_slice = cv2.flip(roi_slice, 1)
        return flipped_study_slice, flipped_roi_slice

    def _apply_and_save_augmentation(self, patient_id: str, augmentation_type: str) -> None:
        # Search for niigz files that include the patient's id in its corresponding name
        # But doesn't include the augmentation method in its name, since we dont want to
        # apply the agumentation technique two times to the same image.
        study_files = self._find_files(self.study_path, patient_id,
                                       exclude_keywords=["flip", "shift", "gamma", "brightness"])
        roi_files = self._find_files(self.roi_path, patient_id,
                                     exclude_keywords=["flip", "shift", "gamma", "brightness"])

        # Asumir que el primer archivo encontrado es el correcto si hay más de uno
        study_file_path = study_files[0] if study_files else None
        roi_file_path = roi_files[0] if roi_files else None

        if not study_file_path or not roi_file_path:
            print(f"Patient {patient_id} is test file or missing.")
            return

        # El resto del código sigue igual...
        study_volume = nib.load(study_file_path)
        study_data = study_volume.get_fdata()
        roi_volume = nib.load(roi_file_path)
        roi_data = roi_volume.get_fdata()

        transformed_study_slices = []
        transformed_roi_slices = []

        for slice_index in self.valid_roi_slices.get(patient_id, []):
            if augmentation_type == "flip":
                transformed_study_slice, transformed_roi_slice = self._flip(study_data[:, :, slice_index],
                                                                            roi_data[:, :, slice_index])
            elif augmentation_type == "shift":
                transformed_study_slice, transformed_roi_slice = self._shift(study_data[:, :, slice_index],
                                                                             roi_data[:, :, slice_index])
            elif augmentation_type == "brightness":
                transformed_study_slice, transformed_roi_slice = self._brightness(study_data[:, :, slice_index],
                                                                                  roi_data[:, :, slice_index])
            elif augmentation_type == "gamma":
                transformed_study_slice, transformed_roi_slice = self._gamma(study_data[:, :, slice_index],
                                                                             roi_data[:, :, slice_index])
            transformed_study_slices.append(transformed_study_slice)
            transformed_roi_slices.append(transformed_roi_slice)
        transformed_study_volume = np.stack(transformed_study_slices,
                                            axis=-1) if transformed_study_slices else np.array([])
        transformed_roi_volume = np.stack(transformed_roi_slices, axis=-1) if transformed_roi_slices else np.array([])

        if transformed_study_volume.any():
            self._save_transformed_volume(patient_id, transformed_study_volume, study_volume.affine, "study",
                                          augmentation_type)
        if transformed_roi_volume.any():
            self._save_transformed_volume(patient_id, transformed_roi_volume, roi_volume.affine, "roi",
                                          augmentation_type)

    def _save_transformed_volume(self, patient_id: str, volume_data: np.ndarray, affine: np.ndarray, volume_type: str, augmentation_type: str) -> None:
        base_file_name = f"{patient_id}_{volume_type}_{augmentation_type}" # Define output_file_name before the try block
        try:
            # Generate a new volume of image with the transformation applied
            new_volume = nib.Nifti1Image(volume_data, affine)
            output_file_path = os.path.join(self.roi_path if volume_type == "roi" else self.study_path, base_file_name)
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            
            # Check for existing augmented files for this patient and augmentation type
            existing_files = glob.glob(f"{output_file_path}_*.nii.gz")
            if existing_files:
                # Extract the highest identifier from existing files
                highest_id = max(int(re.search(r'_(\d+)\.nii\.gz', file).group(1)) for file in existing_files)
                # Increment the identifier for the new file
                new_id = highest_id + 1
            else:
                # If no existing files, start with 0
                new_id = 0
            
            # Construct the new file name with the incremented identifier
            output_file_name = f"{base_file_name}_{new_id}.nii.gz"
            output_file_path = os.path.join(os.path.dirname(output_file_path), output_file_name)
            
            # Save the new volume
            nib.save(new_volume, output_file_path)
        except Exception as e:
            print(f"Failed to save {output_file_name}: {e}")


