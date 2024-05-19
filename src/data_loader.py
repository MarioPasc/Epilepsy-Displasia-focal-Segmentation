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

    
    
    
            


