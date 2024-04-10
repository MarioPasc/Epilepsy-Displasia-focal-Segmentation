
import nibabel as nib
import cv2
import numpy as np
from itertools import chain
import os
import glob
from data_loader import HoldOut
from typing import List


class YOLOv8DatasetGenerator:

    def __init__(self, HoldOutInstance: HoldOut) -> None:
        self.train_set = HoldOutInstance.train_set
        self.val_set = HoldOutInstance.val_set
        self.test_set = HoldOutInstance.test_set
        self.augmentation_methods = ["gamma", "brightness", "flip", "shift"]
        self.exclude = list(range(0, 121)) + list(range(200, 257))

        self.dataset_path = HoldOutInstance.dataset_path
        self.study_name = HoldOutInstance.study_name
        self.ds_yolov8 = os.path.join(self.dataset_path, '..', f"{self.study_name}_yolov8_dataset")
        self.roi_study = HoldOutInstance.roi_study
        # Create all the folders:
        os.makedirs(self.ds_yolov8, exist_ok=True)
        for folder in ["images", "labels"]:
            for set_folder in ["train", "val", "test"]:
                os.makedirs(os.path.join(self.ds_yolov8, folder, set_folder), exist_ok=True)

        # Update the training and validation with the augmented images:
        self._organizeAugmentation()

        
    def _organizeAugmentation(self) -> None:  
        study_path = os.path.join(self.dataset_path, self.study_name)
        patients_train = [patient.split("_")[0] + "*" for patient in self.train_set]
        train_all = list(chain(*[glob.glob(os.path.join(study_path, patientAug)) for patientAug in patients_train]))
        self.train_set = [os.path.basename(item) for item in train_all]

        patients_val = [patient.split("_")[0] + "*" for patient in self.val_set]
        val_all = list(chain(*[glob.glob(os.path.join(study_path, patientAug)) for patientAug in patients_val]))
        self.val_set = [os.path.basename(item) for item in val_all]

    def _contoursYOLO_(self, contours: List[List[int]], height: int, width: int, output_path: str) -> None:
        with open(output_path, 'w') as f:
            for contour in contours:
                normalized = contour.squeeze().astype('float') / np.array([width, height])
                str_contour = ' '.join([f"{coord:.6f}" for coord in normalized.flatten()])
                f.write(f"0 {str_contour}\n")   

    def _generateOutputFormat_(self, niiFileName: str) -> str:
        # The study word is reserved for augmented images:
        # 'sub-00115_acq-T2sel_FLAIR.nii.gz' -> Normal image
        # 'sub-00115_study_gamma_1.nii.gz'   ->  Augmented image
        
        id_patient, augmentation_type = niiFileName.split('_')[0], niiFileName.split('_')[-2]
        # Determine the output file name format based on the augmentation type
        if augmentation_type in self.augmentation_methods:
            # This extracts the augmented id:
            # 'sub-00115_study_gamma_1.nii.gz' -> 1
            # 'sub-00115_study_gamma_3.nii.gz' -> 3
            id_augmentation = niiFileName.strip('.nii.gz').split('_')[-1]
            output_format_image = f"{id_patient}_slice-{{i}}_{augmentation_type}_{id_augmentation}.png"
            output_format_label = f"{id_patient}_slice-{{i}}_{augmentation_type}_{id_augmentation}.txt"
        else:
            output_format_image = f"{id_patient}_slice-{{i}}.png"
            output_format_label = f"{id_patient}_slice-{{i}}.txt"
        return output_format_image, output_format_label    

    def _extractPatientInfo_(self, path:str):
        try:
            nii_volume = nib.load(path)
            nii_data = nii_volume.get_fdata()
            return nii_data
        except Exception as e:
            print(f"Failed to load {path}: {e}")
            return

    def _niiToPNG_(self, niigzFile: str, output_format: str, output_path: str) -> bool:
        # Extract patient info
        study_path = os.path.join(self.dataset_path, self.study_name)
        nii_path = os.path.join(study_path, niigzFile)
        nii_data = self._extractPatientInfo_(path=nii_path)

        if nii_data is None:
            return False

        augmentation_type = niigzFile.split('_')[-2]

        # Now we want to exclude some images to avoid confusing the NN
        # Only exclude these images to niigz files that have more than 100 files and are not augmented
        if nii_data.shape[2] > 100 and augmentation_type not in self.augmentation_methods:
            exclude = self.exclude
        else:
            exclude = []

        for slice in range(nii_data.shape[2]):
            if slice in exclude:
                continue
            slice_data = nii_data[:, :, slice]
            # Normalize the slice data to the range [0, 255]
            slice_normalized = cv2.normalize(slice_data, None, 0, 255, cv2.NORM_MINMAX)
            slice_uint8 = np.uint8(slice_normalized)
            # Construct the output file name
            output_file_name = output_format.format(i=slice)
            output_file_path = os.path.join(output_path, output_file_name)
            # Save the slice as a .png file
            cv2.imwrite(output_file_path, slice_uint8)     

        return True     

    def _niiLabelEncoder_(self, niigzFile: str, output_format: str, output_path: str) -> None:
        roi_path = os.path.join(self.dataset_path, self.roi_study)
        # Now we must find the roi equivalent
        split_by_sub = niigzFile.strip(".nii.gz").split("_")
        id_patient, augmentation_type = split_by_sub[0], split_by_sub[-2]
        if augmentation_type in self.augmentation_methods:
            id_augmentation = split_by_sub[3]
            matching_files = glob.glob(os.path.join(roi_path, f"{id_patient}*{augmentation_type}_{id_augmentation}.nii.gz"))
        else:
            matching_files = glob.glob(os.path.join(roi_path, f"{id_patient}*.nii.gz"))

        if not matching_files:
            print(f"No ROI files found for patient ID: {id_patient}")
            return # Skip this patient if no matching files are found

        roi_niiFile = self._extractPatientInfo_(path=os.path.join(roi_path, matching_files[0]))
        if roi_niiFile is None:
            return
        
        # Check if we need to exclude the ROI slice. This is just prevention, we can't know for certain if 
        # we have any ROI region in within the excluded slices
        if roi_niiFile.shape[2] > 100 and augmentation_type not in self.augmentation_methods:
            exclude = self.exclude
        else:
            exclude = []

        for i in range(roi_niiFile.shape[2]):
            if i in exclude:
                continue

            slice = roi_niiFile[:, :, i].astype(np.uint8)
            contours, _ = cv2.findContours(slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Extract the contours that have at least 3 points : [[x1, y1], [x2, y2], [x3, y3], ...]
            contours = [contour for contour in contours if len(contour) >= 3]
            if contours:
                # Construct the output file name
                output_file_name = output_format.format(i=i)
                if id_patient == "sub-00001": print(f"Image output: {output_file_name}")

                output_file_path = os.path.join(output_path, output_file_name)
                self._contoursYOLO_(contours=contours,
                                    height=slice.shape[0],
                                    width=slice.shape[1],
                                    output_path=output_file_path)
        pass

    def generate_yolo_ds(self) -> None:
        study_path = os.path.join(self.dataset_path, self.study_name)
        for niigz_file in os.listdir(study_path):
            if niigz_file in self.train_set:
                image_output_path = os.path.join(self.ds_yolov8, "images", "train")
                label_output_path = os.path.join(self.ds_yolov8, "labels", "train")
            elif niigz_file in self.val_set:
                image_output_path = os.path.join(self.ds_yolov8, "images", "val")
                label_output_path = os.path.join(self.ds_yolov8, "labels", "val")
            elif niigz_file in self.test_set:
                image_output_path = os.path.join(self.ds_yolov8, "images", "test")
                label_output_path = os.path.join(self.ds_yolov8, "labels", "test")

            format_image, format_label = self._generateOutputFormat_(niiFileName=niigz_file)

            # 1. Save all the images as .png files
            completed = self._niiToPNG_(niigzFile=niigz_file, output_format=format_image, output_path=image_output_path)    
            if completed: # If we can generate the image, generate the corresponding label
                self._niiLabelEncoder_(niigzFile=niigz_file, output_format=format_label, output_path=label_output_path)

