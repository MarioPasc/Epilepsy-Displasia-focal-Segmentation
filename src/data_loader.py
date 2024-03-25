import os
import glob
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional
import shutil

class DataLoader:
    def __init__(self, dataset_path: str) -> None:
        self.dataset_path = dataset_path
        self.patients_roi: List[str] = []
        
    @staticmethod
    def check_patient_for_roi(anat_path: str) -> Optional[str]:
        roi_files = glob.glob(os.path.join(anat_path, '*roi*.nii.gz'))
        if roi_files:
            return os.path.basename(os.path.dirname(anat_path))

    def find_patients_with_roi(self) -> List[str]:
        with ThreadPoolExecutor() as executor:
            futures = []

            for patient_folder in os.listdir(self.dataset_path):
                anat_path = os.path.join(self.dataset_path, patient_folder, 'anat')
                if os.path.isdir(anat_path):
                    futures.append(executor.submit(DataLoader.check_patient_for_roi, anat_path))

            for future in as_completed(futures):
                result = future.result()
                if result:
                    self.patients_roi.append(result)

    def organize_patients_data(self) -> None:
        base_path = os.path.join(self.dataset_path, "..", "ds-epilepsy")
        if (not os.path.exists(base_path)):
            os.makedirs(base_path, exist_ok=True)
            for folder in ["T2FLAIR", "T1WEIGHTED", "ROI"]:
                os.makedirs(os.path.join(base_path, folder), exist_ok=True)

        for patient in self.patients_roi:
            anat_path = os.path.join(self.dataset_path, patient, 'anat')
            for file in os.listdir(anat_path):
                if ('roi' in file.lower() and file.endswith('.nii.gz')):
                    shutil.copy(os.path.join(anat_path, file), os.path.join(base_path, "ROI"))
                elif (('t2' in file.lower() or 'flair' in file.lower()) and file.endswith('.nii.gz')):
                    shutil.copy(os.path.join(anat_path, file), os.path.join(base_path, "T2FLAIR"))
                elif ('t1' in file.lower() and file.endswith('.nii.gz')):
                    shutil.copy(os.path.join(anat_path, file), os.path.join(base_path, "T1WEIGHTED"))


class HoldOut:
    def __init__(self, dataset_path: str) -> None:
        self.train_set: List[str] = []
        self.val_set: List[str] = []
        self.test_set: List[str] = []
        self.dataset_path = dataset_path

    def holdout(self, study_name: str, val_percent: float, test_percent: float) -> None:
        self.train_set.clear()
        self.val_set.clear()
        self.test_set.clear()
        # Access the .nii.gz study files
        folder_path = os.path.join(self.dataset_path, study_name)
        niigz_files = [os.path.join(study_name, file) for file in os.listdir(folder_path) if file.endswith('.nii.gz')]
        # Split
        val_test_size = val_percent + test_percent
        train_files, val_test_files = train_test_split(niigz_files, test_size=val_test_size, random_state=42)
        val_files, test_files = train_test_split(val_test_files, test_size=test_percent / val_test_size, random_state=42)
        
        self.train_set.extend(train_files)
        self.val_set.extend(val_files)
        self.test_set.extend(test_files)
        
        # Check wether if the sum of all *_set files is the same as the study files
        
        if len(niigz_files) == len(self.train_set) + len(self.val_set) + len(self.test_set):
            print("Hold-out ended successfully.")
        else:
            print(f"Hold-out error. The total file amount is {len(niigz_files)}" + 
                    "and the sum of all sets is {len(self.train_set) + len(self.val_set) + len(self.test_set)}.")

        
    

def main():
    dl_instance = DataLoader(dataset_path="/home/mario/VSCode/Dataset/epilepsy")
    dl_instance.find_patients_with_roi()
    dl_instance.organize_patients_data()
    holdout_instance = HoldOut(dataset_path="/home/mario/VSCode/Dataset/ds-epilepsy")
    holdout_instance.holdout(study_name="T2FLAIR",
                             val_percent=.2,
                             test_percent=.1)
    print(f"Train: {holdout_instance.train_set}\nVal: {holdout_instance.val_set}\nTest: {holdout_instance.test_set}")
    

if __name__=="__main__":
    main()

