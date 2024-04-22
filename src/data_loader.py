import os
import glob
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional
import shutil


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
        patients_roi = []
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(self.__check_patient_for_roi__, os.path.join(self.dataset_path, patient_folder, 'anat'))
                       for patient_folder in os.listdir(self.dataset_path)
                       if os.path.isdir(os.path.join(self.dataset_path, patient_folder, 'anat'))}

            for future in as_completed(futures):
                result = future.result()
                if result:
                    patients_roi.append(result)
        self.patients_roi = patients_roi

    def __organize_patients_data__(self) -> None:
        base_path = os.path.join(self.dataset_path, "..", "ds-epilepsy")
        if (not os.path.exists(base_path)):
            os.makedirs(base_path, exist_ok=True)
            for folder in ["T2FLAIR", "T1WEIGHTED", "ROI_T2", "ROI_T1"]:
                study_folder = os.path.join(base_path, folder)
                os.makedirs(study_folder, exist_ok=True)

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
        self.dataset_path = dataset_path

        self.val_percent = val_percent
        self.test_percent = test_percent
        self.__holdoutNiigz__()
    
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

    """
    # We want to relocate the files in train test val folders before generating the final dataset
    def __relocateFiles__(self):
        study_path = os.path.join(self.dataset_path, self.study_name)
        for setFolder in ["train", "val", "test"]:
            os.makedirs(os.path.join(study_path, setFolder), exist_ok=True)

        for file in os.listdir(study_path):
            niigz_filePath = os.path.join(study_path, file)
            if file in self.train_set:
                shutil.move(niigz_filePath, os.path.join(study_path, "train"))
            elif file in self.val_set:
                shutil.move(niigz_filePath, os.path.join(study_path, "val"))
            elif file in self.test_set:
                shutil.move(niigz_filePath, os.path.join(study_path, "test")) """

    
    
    
            


