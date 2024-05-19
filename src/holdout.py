import os
from sklearn.model_selection import train_test_split
from typing import List

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
            
    def _writeHoldOutInfo_(self) -> None:
        """Write the hold out information to separate text files."""
        # Create the directory if it doesn't exist
        os.makedirs("./txt", exist_ok=True)

        # Write the train set to a text file
        with open("./txt/train_set.txt", "w") as file:
            file.write("\n".join(self.train_set))

        # Write the validation set to a text file
        with open("./txt/val_set.txt", "w") as file:
            file.write("\n".join(self.val_set))

        # Write the test set to a text file
        with open("./txt/test_set.txt", "w") as file:
            file.write("\n".join(self.test_set))
