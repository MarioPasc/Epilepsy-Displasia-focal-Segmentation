import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO
import numpy as np
import os
import glob
import csv
import cv2
from typing import List
import nibabel as nib
import torch


class TestYoloV8:

    def __init__(self, weights_folder:str, dataset_path:str, test_results_path:str, roi_study_path:str, study_path:str,
                 output_dir:str) -> None:
        self.weights_folder = weights_folder
        self.dataset_path = dataset_path
        self.test_results_path = test_results_path
        self.roi_study_path = roi_study_path
        self.study_path = study_path
        self.output_dir = output_dir
        
        self.masks = {}
        self.stats_df = pd.DataFrame(columns=["patient_id", "slice", "exist", "dice_score"])

    def test_yolov8(self) -> None:
        predict_files = os.listdir(os.path.join(self.dataset_path, "images/test"))
        predict_files = [os.path.join(self.dataset_path, "images/test", file) for file in predict_files]
        predict_files = [file for file in predict_files if os.path.exists(file)]
        best_pt = os.path.join(self.weights_folder, "best.pt")
        model = YOLO(best_pt)

        for i, file in enumerate(predict_files):
            results = model(file, stream=True)
            thisFileName = os.path.basename(file).strip(".png").split("-")
            patient_id = int(thisFileName[1].split("_")[0])
            slice = int(thisFileName[2])

            for im_pred in results:
                mask = im_pred.masks
                if mask is not None:
                    self.masks[os.path.basename(file)] = mask.cpu()
                    new_row = pd.DataFrame({"patient_id": [patient_id], "slice": [slice], "exist": [1], "dice_score": [0]})
                    self.stats_df = pd.concat([self.stats_df, new_row], ignore_index=True)
                else:
                    new_row = pd.DataFrame({"patient_id": [patient_id], "slice": [slice], "exist": [0], "dice_score": [0]})
                    self.stats_df = pd.concat([self.stats_df, new_row], ignore_index=True)
                im_pred.save(filename=os.path.join(self.test_results_path, f"{os.path.basename(file)}_predict.png"))

    def _obtain_test_patients(self) -> List[str]:
        test_folder = os.path.join(self.dataset_path, "labels/test")
        txt_files = glob.glob(os.path.join(test_folder, "*.txt"))
        
        subject_ids = []
        
        # Iterar sobre cada archivo .txt
        for file_path in txt_files:
            # Obtener el nombre del archivo sin la extensión
            file_name = os.path.basename(file_path)
            
            # Extraer el identificador 'sub-00XXX' del nombre del archivo
            subject_id = file_name.split("_")[0]
            
            # Agregar el identificador a la lista si no está duplicado
            if subject_id not in subject_ids:
                subject_ids.append(subject_id)
        return subject_ids

    def _dice_score(self,mask1, mask2) -> float:
        intersection = np.sum((mask1 > 0) & (mask2 > 0))
        total = np.sum(mask1 > 0) + np.sum(mask2 > 0)
        if total == 0:
            return 1.0  # Si ambas máscaras están vacías, el score es perfecto
        return 2 * intersection / total

    def _analize_and_plot(self, patient_id, slice_number, image_niigz_file, roi_niigz_file, masks, kernel_size=3, iterations=2) -> None:
        merged_tensor = np.zeros(shape=(masks.shape[1], masks.shape[2]))
        for i in range(masks.shape[0]):
            merged_tensor += masks[i]
        merged_tensor = np.clip(merged_tensor, 0, 1)

        # Load the NIfTI files
        image_data = nib.load(image_niigz_file).get_fdata()
        roi_data = nib.load(roi_niigz_file).get_fdata()
        # Get the corresponding slice
        image_slice = image_data[:, :, slice_number]
        roi_slice = roi_data[:, :, slice_number]
        
        roi_slice = cv2.resize(roi_slice, (merged_tensor.shape[1], merged_tensor.shape[0]), interpolation=cv2.INTER_AREA)
        image_slice = cv2.resize(image_slice, (merged_tensor.shape[1], merged_tensor.shape[0]), interpolation=cv2.INTER_AREA)

        # Apply erosion to the ground truth mask and the corresponding region in the image
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        eroded_roi_slice = cv2.erode(roi_slice, kernel, iterations=iterations)
        eroded_roi_slice = cv2.threshold(eroded_roi_slice, 0, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)
    
        eroded_image_slice = cv2.bitwise_and(image_slice, image_slice, mask=eroded_roi_slice)

        # Calculate the Dice score 
        dice = self._dice_score(eroded_roi_slice, merged_tensor)
        # Update the dice_score in self.stats_df
        mask_stat = (self.stats_df['patient_id'] == int(patient_id.split("-")[1])) & (self.stats_df['slice'] == int(slice_number))
        self.stats_df.loc[mask_stat, 'dice_score'] = dice
        # Create the visualization
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        ax1.imshow(image_slice, cmap='gray')
        ax1.set_title('Image')
        ax1.axis('off')
        
        ax2.imshow(eroded_image_slice, cmap='gray')
        ax2.imshow(eroded_roi_slice, cmap='jet', alpha=0.7)
        ax2.set_title('Eroded Original Mask')
        ax2.axis('off')
        
        ax3.imshow(image_slice, cmap='gray')
        ax3.imshow(merged_tensor, cmap='jet', alpha=0.7)
        ax3.set_title('Predicted Mask')
        ax3.axis('off')
        
        fig.suptitle(f'Patient: {patient_id}, Slice: {slice_number}, Dice Score: {dice:.4f}')
        fig.tight_layout()
        
        # Save the image in the current directory
        os.makedirs(os.path.join(self.test_results_path, "test_check"), exist_ok=True)
        plt.savefig(os.path.join(os.path.join(self.test_results_path, "test_check"), f'{patient_id}_slice_{slice_number}_comparison.png'))
        plt.close(fig)

    def check_test_results(self) -> None:
        test_patients = self._obtain_test_patients()
        # Obtain the niigz test files
        roi_test_niigzFiles = [test_files for test_files in os.listdir(self.roi_study_path) if test_files.split("_")[0] in test_patients]
        image_test_niigzFiles = [test_files for test_files in os.listdir(self.study_path) if test_files.split("_")[0] in test_patients]


        for pred_patient in self.masks.keys():
            # Obtain the slice number from the patient file name
            slice_number = int(pred_patient.strip(".png").split("-")[-1])
            # Obtain the corresponding mask
            patient_id = pred_patient.split("_")[0]
            roi_test_niigzFile = [file for file in roi_test_niigzFiles if file.startswith(patient_id)]
            image_test_niigzFile = [file for file in image_test_niigzFiles if file.startswith(patient_id)]
            if len(roi_test_niigzFile) != 0 and len(image_test_niigzFile) != 0: # If we have found a test ROI
                # Get image data
                image_niigz_file = os.path.join(self.study_path, image_test_niigzFile[0])
                # Get ground truth data
                roi_niigz_file = os.path.join(self.roi_study_path, roi_test_niigzFile[0])
                # Get prediction data
                mask = self.masks.get(pred_patient).data.cpu().numpy()

                self._analize_and_plot(patient_id=patient_id, slice_number=slice_number, image_niigz_file=image_niigz_file, roi_niigz_file=roi_niigz_file, masks=mask)

                torch.save(mask, os.path.join(self.output_dir, "maskTensors", f"{patient_id}_slice_{slice_number}.pt"))                
                # Save self.masks.get(pred_patient).xy to a .txt file
                output_file = os.path.join(self.output_dir, "xyn_coordinates", f"{patient_id}_slice_{slice_number}.txt")
                with open(output_file, "w") as f:
                    xyn_coords = self.masks.get(pred_patient).xyn
                    
                    for contour in xyn_coords:
                        # Format the XY coordinates as a string with numbers separated by spaces
                        xyn_coords_str = " ".join([f"{coord[0]} {coord[1]}" for coord in contour])
                        
                        # Add a "0" before each set of XY coordinates
                        f.write("0 ")
                        f.write(xyn_coords_str)
                        f.write("\n")
                self.stats_df.sort_values(by="patient_id")
                self.stats_df.to_csv(os.path.join(self.output_dir, "stats.csv"))



def main() -> None:
    path_results = "/home/mariopasc/Python/Results"
    study = "Kernel3Iterations2"
    path_study_to_test = os.path.join(path_results, study)
    results = os.path.join(path_study_to_test, "Results-BiggerFCD")
    testYOLO = TestYoloV8(weights_folder=os.path.join(results, "weights"),
                          test_results_path=os.path.join(results, "test"),
                          dataset_path= os.path.join(path_study_to_test, "kernelsize3iterations2"),
                          roi_study_path="/home/mariopasc/Python/Datasets/ds-epilepsy/BiggerFCD_ROIT2FLAIR",
                          study_path="/home/mariopasc/Python/Datasets/ds-epilepsy/BiggerFCD_T2FLAIR",
                          output_dir=results)
    

    #ValInstance.validateYolov8()
    #ValInstance.val_metrics()
    testYOLO.test_yolov8()
    testYOLO.check_test_results()

if __name__ == "__main__":
    main()