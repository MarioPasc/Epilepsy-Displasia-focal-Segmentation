import os
import glob
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional
import shutil
import nibabel as nib
import cv2
import numpy as np
from tqdm import tqdm
from itertools import chain

class DataLoader:
    def __init__(self, dataset_path: str) -> None:
        self.dataset_path = dataset_path
        self.patients_roi: List[str] = []
        self.__find_patients_with_roi__()
        self.__organize_patients_data__()

    @staticmethod
    def __check_patient_for_roi__(anat_path: str) -> Optional[str]:
        roi_files = glob.glob(os.path.join(anat_path, '*roi*.nii.gz'))
        if roi_files:
            return os.path.basename(os.path.dirname(anat_path))

    def __find_patients_with_roi__(self) -> List[str]:
        with ThreadPoolExecutor() as executor:
            futures = []

            for patient_folder in os.listdir(self.dataset_path):
                anat_path = os.path.join(self.dataset_path, patient_folder, 'anat')
                if os.path.isdir(anat_path):
                    futures.append(executor.submit(DataLoader.__check_patient_for_roi__, anat_path))

            for future in as_completed(futures):
                result = future.result()
                if result:
                    self.patients_roi.append(result)

    def __organize_patients_data__(self) -> None:
        base_path = os.path.join(self.dataset_path, "..", "ds-epilepsy")
        if (not os.path.exists(base_path)):
            os.makedirs(base_path, exist_ok=True)
            for folder in ["T2FLAIR", "T1WEIGHTED", "ROI_T2", "ROI_T1"]:
                study_folder = os.path.join(base_path, folder)
                os.makedirs(study_folder, exist_ok=True)
                os.makedirs(os.path.join(study_folder, "TEST"), exist_ok=True)

        for patient in self.patients_roi:
            anat_path = os.path.join(self.dataset_path, patient, 'anat')
            for file in os.listdir(anat_path):
                if ('roi' in file.lower() and file.endswith('.nii.gz')):
                    shutil.copy(os.path.join(anat_path, file), os.path.join(base_path, "ROI_T1"))
                elif (('t2' in file.lower() or 'tse3dvfl' in file.lower()) and file.endswith('.nii.gz')):
                    shutil.copy(os.path.join(anat_path, file), os.path.join(base_path, "T2FLAIR"))
                elif ('t1' in file.lower() and file.endswith('.nii.gz')):
                    shutil.copy(os.path.join(anat_path, file), os.path.join(base_path, "T1WEIGHTED"))
            for file in os.listdir(os.path.join(base_path, "ROI_T1")):
                if ('t2' in file.lower() or 'tse3dvfl' in file.lower()):
                    shutil.copy(os.path.join(base_path, "ROI_T1", file), os.path.join(base_path, "ROI_T2"))


class HoldOut:
    def __init__(self, dataset_path: str, study_name: str, roi_study: str, val_percent: float,
                 test_percent: float) -> None:
        self.train_set: List[str] = []
        self.val_set: List[str] = []
        self.test_set: List[str] = []

        self.roi_study = roi_study
        self.study_name = study_name
        self.val_percent = val_percent
        self.test_percent = test_percent
        self.dataset_path = dataset_path
        self.__holdoutNiigz__()
        self.__relocateTestFiles__()
        

    def __holdoutNiigz__(self) -> None:
        self.train_set.clear()
        self.val_set.clear()
        self.test_set.clear()
        # Access the .nii.gz study files
        folder_path = os.path.join(self.dataset_path, self.study_name)
        niigz_files = [file for file in os.listdir(folder_path) if file.endswith('.nii.gz')]
        # Split
        val_test_size = self.val_percent + self.test_percent
        train_files, val_test_files = train_test_split(niigz_files, test_size=val_test_size, random_state=3)
        val_files, test_files = train_test_split(val_test_files, test_size=self.test_percent / val_test_size,
                                                 random_state=3)

        self.train_set.extend(train_files)
        self.val_set.extend(val_files)
        self.test_set.extend(test_files)

        # Check wether if the sum of all *_set files is the same as the study files

        if len(niigz_files) == len(self.train_set) + len(self.val_set) + len(self.test_set):
            print("Hold-out ended successfully.")
        else:
            print(f"Hold-out error. The total file amount is {len(niigz_files)}" +
                  "and the sum of all sets is {len(self.train_set) + len(self.val_set) + len(self.test_set)}.")

    def __relocateTestFiles__(self) -> None:
        study_path = os.path.join(self.dataset_path, self.study_name)
        for file in os.listdir(study_path):
            if file in self.test_set:
                shutil.move(src=os.path.join(study_path, file),
                            dst=os.path.join(study_path, "TEST"))

    def __niiPng__(self, nii_path: str, output_path: str) -> None:
        try:
            nii_volume = nib.load(nii_path)
            nii_data = nii_volume.get_fdata()
        except Exception as e:
            print(f"Failed to load {nii_path}: {e}")
            return
        
        # Extract patient ID and augmentation type from the file name
        file_name = os.path.basename(nii_path)
        id_patient, augmentation_type = file_name.split('_')[0], file_name.split('_')[-2]
        # Determine the output file name format based on the augmentation type
        if augmentation_type in ["gamma", "brightness", "flip", "shift"]:
            id_augmentation = file_name.split('.')[0].split('_')[-1]
            output_format = f"{id_patient}_slice-{{i}}_{augmentation_type}_{id_augmentation}.png"
        else:
            output_format = f"{id_patient}_slice-{{i}}.png"


        # Convert each slice to a .png file
        exclude = list(range(0, 121)) + list(range(200, 257))
        for i in range(nii_data.shape[2]):
            if output_format == f"{id_patient}_slice-{{i}}.png":
                # Exclude only images that have more than a hundred images per patient
                if i in exclude and nii_data.shape[2] > 100:
                    continue
            slice_data = nii_data[:, :, i]
            # Normalize the slice data to the range [0, 255]
            slice_normalized = cv2.normalize(slice_data, None, 0, 255, cv2.NORM_MINMAX)
            slice_uint8 = np.uint8(slice_normalized)
            # Construct the output file name
            output_file_name = output_format.format(i=i)
            output_file_path = os.path.join(output_path, output_file_name)
            # Save the slice as a .png file
            cv2.imwrite(output_file_path, slice_uint8)        
            
                
    def __roiContours__(self, nii_path: str, output_path: str) -> None:
        roi_path = os.path.join(self.dataset_path, self.roi_study)
        # Determine the output file name format based on the augmentation type
        # Extract patient ID and augmentation type from the file name
        file_name = os.path.basename(nii_path)
        id_patient, augmentation_type = file_name.split('_')[0], file_name.split('_')[-2]
        # Determine the output file name format based on the augmentation type
        if augmentation_type in ["gamma", "brightness", "flip", "shift"]:
            id_augmentation = file_name.split('.')[0].split('_')[-1]
            output_format = f"{id_patient}_slice-{{i}}_{augmentation_type}_{id_augmentation}.txt"
            matching_files = glob.glob(os.path.join(roi_path, f"{id_patient}*{augmentation_type}_{id_augmentation}.nii.gz"))
        else:
            output_format = f"{id_patient}_slice-{{i}}.txt"    
            matching_files = glob.glob(os.path.join(roi_path, f"{id_patient}*.nii.gz"))

        
        if not matching_files:
            print(f"No ROI files found for patient ID: {id_patient}")
            return # Skip this patient if no matching files are found

        file = os.path.join(roi_path, matching_files[0])    

        if id_patient == "sub-00001": print(f"Image: {nii_path} \nMatching: {file}")  
        
        roi_niigz = nib.load(os.path.join(roi_path, file)).get_fdata()
        exclude = list(range(0, 121)) + list(range(200, 257))
        for i in range(roi_niigz.shape[2]):
            if output_format == f"{id_patient}_slice-{{i}}.txt":
                # Exclude only images that have more than a hundred images per patient
                if i in exclude and roi_niigz.shape[2] > 100:
                    continue
            slice = roi_niigz[:, :, i].astype(np.uint8)
            contours, _ = cv2.findContours(slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Extract the contours that have at least 3 points : [[x1, y1], [x2, y2], [x3, y3], ...]
            contours = [contour for contour in contours if len(contour) >= 3]
            if contours:
                

                  
                # Construct the output file name
                output_file_name = output_format.format(i=i)
                output_file_path = os.path.join(output_path, output_file_name)
                self.__contoursYOLO__(contours=contours,
                                      height=slice.shape[0],
                                      width=slice.shape[1],
                                      output_path=output_file_path)

    def __contoursYOLO__(self, contours: List[List[int]], height: int, width: int, output_path: str) -> None:
        with open(output_path, 'w') as f:
            for contour in contours:
                normalized = contour.squeeze().astype('float') / np.array([width, height])
                str_contour = ' '.join([f"{coord:.6f}" for coord in normalized.flatten()])
                f.write(f"0 {str_contour}\n")

    def organizeAugmentation(self) -> None:  
        study_path = os.path.join(self.dataset_path, self.study_name)
        patients_train = [patient.split("_")[0] + "*" for patient in self.train_set]
        train_all = list(chain(*[glob.glob(os.path.join(study_path, patientAug)) for patientAug in patients_train]))
        self.train_set = [os.path.basename(item) for item in train_all]

        patients_val = [patient.split("_")[0] + "*" for patient in self.val_set]
        val_all = list(chain(*[glob.glob(os.path.join(study_path, patientAug)) for patientAug in patients_val]))
        self.val_set = [os.path.basename(item) for item in val_all]

        

    def holdout(self) -> None:
        # Create folder structure
        base_path = os.path.join(self.dataset_path, "..", f"{self.study_name}-ds-epilepsy")
        if (not os.path.exists(base_path)):
            os.makedirs(base_path, exist_ok=True)
            for data_type in ["images", "labels"]:
                for folder in ["train", "val", "test"]:
                    os.makedirs(os.path.join(base_path, data_type, folder), exist_ok=True)
        # 1. Copy niigz files as png in their respective holdout folder
        # 2. Extract ROI contours and save them in their respective holdout folder
        study_path = os.path.join(self.dataset_path, self.study_name)
        nifti_files = os.listdir(study_path)
        for niigz_file in tqdm(nifti_files, desc="Generating YOLO dataset"):
            nii_path = os.path.join(study_path, niigz_file)
            
            if niigz_file in self.train_set:
                image_output_path = os.path.join(base_path, "images", "train")
                label_output_path = os.path.join(base_path, "labels", "train")
            elif niigz_file in self.val_set:
                image_output_path = os.path.join(base_path, "images", "val")
                label_output_path = os.path.join(base_path, "labels", "val")
            else:
                continue

            self.__niiPng__(nii_path=nii_path, output_path=image_output_path)
            self.__roiContours__(nii_path=nii_path, output_path=label_output_path)

        for test_image in os.listdir(os.path.join(study_path, "TEST")):
            image_output_path = os.path.join(base_path, "images", "test")
            label_output_path = os.path.join(base_path, "labels", "test")
            nii_path = os.path.join(study_path, "TEST", test_image)
            self.__niiPng__(nii_path=nii_path, output_path=image_output_path)
            self.__roiContours__(nii_path=nii_path, output_path=label_output_path)


