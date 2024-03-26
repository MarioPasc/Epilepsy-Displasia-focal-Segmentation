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
import math

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
                os.makedirs(os.path.join(base_path, folder), exist_ok=True)

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
    def __init__(self, dataset_path: str, study_name: str, roi_study: str, val_percent: float, test_percent: float) -> None:
        self.train_set: List[str] = []
        self.val_set: List[str] = []
        self.test_set: List[str] = []
        
        self.roi_study = roi_study
        self.study_name = study_name
        self.val_percent = val_percent
        self.test_percent = test_percent
        self.dataset_path = dataset_path
        self.__holdoutNiigz__()     
        self.__holdout__()  
        

    def __holdoutNiigz__(self) -> None:
        self.train_set.clear()
        self.val_set.clear()
        self.test_set.clear()
        # Access the .nii.gz study files
        folder_path = os.path.join(self.dataset_path, self.study_name)
        niigz_files = [file for file in os.listdir(folder_path) if file.endswith('.nii.gz')]
        # Split
        val_test_size = self.val_percent + self.test_percent
        train_files, val_test_files = train_test_split(niigz_files, test_size=val_test_size, random_state=42)
        val_files, test_files = train_test_split(val_test_files, test_size=self.test_percent / val_test_size, random_state=42)
        
        self.train_set.extend(train_files)
        self.val_set.extend(val_files)
        self.test_set.extend(test_files)
        
        # Check wether if the sum of all *_set files is the same as the study files
        
        if len(niigz_files) == len(self.train_set) + len(self.val_set) + len(self.test_set):
            print("Hold-out ended successfully.")
        else:
            print(f"Hold-out error. The total file amount is {len(niigz_files)}" + 
                    "and the sum of all sets is {len(self.train_set) + len(self.val_set) + len(self.test_set)}.")
            
    def __niiPng__(self, nii_path: str, output_path: str) -> None:
        nii_volume = nib.load(nii_path)
        nii_data = nii_volume.get_fdata()
        id_patient = (nii_path.split('/')[-1]).split("_")[0]
        exclude = list(range(0, 121)) + list(range(220, 257))
        for i in range(nii_data.shape[2]):
            if i not in exclude:
                # Temporal normalization
                slice_normalized = cv2.normalize(nii_data[:, :, i], None, 0, 255, cv2.NORM_MINMAX)
                slice_uint8 = np.uint8(slice_normalized)
                cv2.imwrite(os.path.join(output_path, f'{id_patient}_slice-{i}.png'), slice_uint8)
    
    def __contoursYOLO__(self, contours: List[List[int]], height: int, width: int, output_path: str):
            with open(output_path, 'w') as f:
                for contour in contours:
                    normalized = contour.squeeze().astype('float') / np.array([width, height])
                    str_contour = ' '.join([f"{coord:.6f}" for coord in normalized.flatten()])
                    f.write(f"0 {str_contour}\n")

    def __roiContours__(self, nii_path: str, output_path: str):
        roi_path = os.path.join(self.dataset_path, self.roi_study)
        # Obtain the corresponding ROI niigz file
        id_patient = (nii_path.split('/')[-1]).split("_")[0]
        file = os.path.join(roi_path, glob.glob(os.path.join(roi_path, f"{id_patient}*.nii.gz"))[0])
        roi_niigz = nib.load(os.path.join(roi_path, file)).get_fdata()
        for i in range(roi_niigz.shape[2]):
            slice = roi_niigz[:, :, i].astype(np.uint8)
            contours, _ = cv2.findContours(slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Extract the contours that have at least 3 points : [[x1, y1], [x2, y2], [x3, y3], ...]
            contours = [contour for contour in contours if len(contour) >= 3]  
            if contours: 
                self.__contoursYOLO__(contours=contours,
                                    height=slice.shape[0],
                                    width=slice.shape[1],
                                    output_path=os.path.join(output_path, f"{id_patient}_slice-{i}.txt"))

    def __holdout__(self):
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
            elif niigz_file in self.test_set:
                image_output_path = os.path.join(base_path, "images", "test")
                label_output_path = os.path.join(base_path, "labels", "test")
            else:
                continue
            
            self.__niiPng__(nii_path=nii_path, output_path=image_output_path)
            self.__roiContours__(nii_path=nii_path,output_path=label_output_path)

    

def main():
    dl_instance = DataLoader(dataset_path="/home/mario/VSCode/Dataset/epilepsy")
    holdout_instance = HoldOut(dataset_path="/home/mario/VSCode/Dataset/ds-epilepsy",
                               study_name="T2FLAIR", roi_study = "ROI_T2", val_percent=.2, test_percent=.1)    
    

if __name__=="__main__":
    main()

