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



class localYoloV8:

    def __init__(self, train_csv_path: str, val_output_path: str, weights_folder:str, 
                 yaml_file:str, dataset_path:str, test_results_path:str, roi_study_path:str, study_path:str,
                 output_dir:str) -> None:
        self.train_csv_path = train_csv_path
        self.weights_folder = weights_folder
        self.val_output_path = val_output_path
        self.yaml_file = yaml_file
        self.dataset_path = dataset_path
        self.test_results_path = test_results_path
        self.roi_study_path = roi_study_path
        self.study_path = study_path
        self.output_dir = output_dir
        
        self.masks = {}

    def validateYolov8(self):
        weights_files = sorted([file for file in glob.glob(os.path.join(self.weights_folder, '*.pt'))
                                if 'last.pt' not in file])
        # Verificar si 'last.pt' existe y agregarlo al final de la lista
        last_pt_path = os.path.join(self.weights_folder, 'last.pt')
        if os.path.exists(last_pt_path):
            weights_files.append(last_pt_path)

        epoch_counter = 1 
        with open(self.val_output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['epoch', 'file',
                             'precision(B)', 'recall(B)', 'mAP50(B)', 'mAP50-95(B)',
                             'precision(M)', 'recall(M)', 'mAP50(M)', 'mAP50-95(M)',
                             'TP', 'TN', 'FP', 'FN'])

            for weight_file in weights_files:
                # Identificar si el archivo es 'best.pt' o 'last.pt'
                if 'best.pt' in weight_file:
                    epoch = 101
                    file_name = 'best'
                elif 'last.pt' in weight_file:
                    epoch = 100  # Usamos 'N' como marcador para 'last.pt'
                    file_name = 'last'
                else:
                    epoch = epoch_counter
                    file_name = f"epoch{epoch_counter}"
                    epoch_counter += 1  # Incrementar para la próxima época

                model = YOLO(model=weight_file, task='segment')
                results = model.val(data=self.yaml_file, imgsz=640, conf=0.01, plots=True)

                precision_b = results.box.mp  # precisión media
                recall_b = results.box.mr  # recall medio
                map_05_b = results.box.map50  # mAP para th=.5
                map_05_95_b = results.box.map  # mAP para th=.5-.95

                precision_m = results.seg.mp  # precisión media
                recall_m = results.seg.mr  # recall medio
                map_05_m = results.seg.map50  # mAP para th=.5
                map_05_95_m = results.seg.map  # mAP para th=.5-.95

                conf_mat = results.confusion_matrix.matrix
                tp = conf_mat[0][0]
                TN = conf_mat[1][1]
                FP = conf_mat[0][1]
                FN = conf_mat[1][0]

                writer.writerow([epoch, file_name,
                                 precision_b, recall_b, map_05_b, map_05_95_b,
                                 precision_m, recall_m, map_05_m, map_05_95_m,
                                 tp, TN, FP, FN])
    def val_metrics(self):
        """
        This function plots the training metrics versus validation metrics for each epoch
        using data from 'results.csv' and 'results_val.csv'.
        """
        # Load training and validation metrics
        train_metrics_df = pd.read_csv(self.train_csv_path)
        train_metrics_df.columns = train_metrics_df.columns.str.strip()  
        val_metrics_df = pd.read_csv(self.val_output_path)

        def plot_metric(train_df, val_df, metric_name):
            """
            Plots a specified metric for both training and validation datasets over epochs.
            """
            # Find common epochs to ensure alignment in plots
            common_epochs = pd.Series(list(set(train_df['epoch']) & set(val_df['epoch'])))
            train_df_filtered = train_df[train_df['epoch'].isin(common_epochs)]
            val_df_filtered = val_df[val_df['epoch'].isin(common_epochs)]

            plt.figure(figsize=(10, 5))
            plt.plot(train_df_filtered['epoch'], train_df_filtered[[f"metrics/{metric_name}"]], label=f'Training {metric_name}')
            plt.plot(val_df_filtered['epoch'], val_df_filtered[metric_name], label=f'Validation {metric_name}')
            plt.xlabel('Epoch')
            plt.ylabel(metric_name)
            plt.title(f'Training vs Validation {metric_name} per Epoch')
            plt.legend()

            # Ensure the save path exists
            save_path = "/home/mariopasc/Python/Projects/BSC_final/epilepsy-displasia-focal-segmentation/images"
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(f"{save_path}/{metric_name.replace('metrics/', '')}_train_vs_val_per_epoch.png")
            plt.close()

        # Define the metrics you want to plot
        metric_names = ['precision(M)', 'recall(M)', 'mAP50(M)', 'mAP50-95(M)',
                        'precision(B)', 'recall(B)', 'mAP50(B)', 'mAP50-95(B)']

        for metric_name in metric_names:
            plot_metric(train_metrics_df, val_metrics_df, metric_name)

    def test_yolov8(self):
        predict_files = os.listdir(os.path.join(self.dataset_path, "images/test"))
        predict_files = [os.path.join(self.dataset_path, "images/test", file) for file in predict_files]
        predict_files = [file for file in predict_files if os.path.exists(file)]
        best_pt = os.path.join(self.weights_folder, "best.pt")
        model = YOLO(best_pt)

        for i, file in enumerate(predict_files):
            results = model(file, stream=True)

            for im_pred in results:
                boxes = im_pred.boxes
                probs = im_pred.probs
                mask = im_pred.masks
                if mask is not None: 
                    self.masks[os.path.basename(file)] = mask.cpu()
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

    def _dice_score(self,mask1, mask2):
        intersection = np.sum((mask1 > 0) & (mask2 > 0))
        total = np.sum(mask1 > 0) + np.sum(mask2 > 0)
        if total == 0:
            return 1.0  # Si ambas máscaras están vacías, el score es perfecto
        return 2 * intersection / total


    def _merge_mask_channels(self, mask_tensor):
        # Sum all the channels along the channel dimension
        merged_mask = np.sum(mask_tensor, axis=0)
        
        # Normalize the merged mask to the range [0, 1]
        merged_mask = merged_mask / np.max(merged_mask)
        
        return merged_mask

    def _analize_and_plot(self, patient_id, slice_number, image_niigz_file, roi_niigz_file, mask):
        # Load the NIfTI files
        image_data = nib.load(image_niigz_file).get_fdata()
        roi_data = nib.load(roi_niigz_file).get_fdata()
        
        # Get the corresponding slice
        image_slice = image_data[:, :, slice_number]
        roi_slice = roi_data[:, :, slice_number]
        
        # Merge the channels of the predicted mask
        merged_mask = self._merge_mask_channels(mask)
        
        # Resize the merged mask to match the shape of roi_slice
        mask_resized = cv2.resize(merged_mask, (roi_slice.shape[1], roi_slice.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Calculate the Dice score 
        dice = self._dice_score(roi_slice, mask_resized)
        
        # Create the visualization
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        ax1.imshow(image_slice, cmap='gray')
        ax1.set_title('Image')
        ax1.axis('off')
        
        ax2.imshow(image_slice, cmap='gray')
        ax2.imshow(roi_slice, cmap='jet', alpha=0.7)
        ax2.set_title('Original Mask')
        ax2.axis('off')
        
        ax3.imshow(image_slice, cmap='gray')
        ax3.imshow(mask_resized, cmap='jet', alpha=0.7)
        ax3.set_title('Predicted Mask')
        ax3.axis('off')
        
        fig.suptitle(f'Patient: {patient_id}, Slice: {slice_number}, Dice Score: {dice:.4f}')
        fig.tight_layout()
        
        # Save the image in the current directory
        os.makedirs(os.path.join(self.test_results_path, "test_check"), exist_ok=True)
        plt.savefig(os.path.join(os.path.join(self.test_results_path, "test_check"), f'{patient_id}_slice_{slice_number}_comparison.png'))
        plt.close(fig)

    def visualize_mask_tensor(self, mask_tensor, patient_id, slice_number, save_path):
        # Determine the number of channels in the mask tensor
        num_channels = mask_tensor.shape[0]
        
        # Create a figure with subplots for each channel
        fig, axes = plt.subplots(1, num_channels, figsize=(5 * num_channels, 5))
        
        # If there is only one channel, make sure axes is a list
        if num_channels == 1:
            axes = [axes]
        
        # Iterate over each channel and plot the mask
        for i in range(num_channels):
            axes[i].imshow(mask_tensor[i], cmap='gray')
            axes[i].set_title(f'Mask Channel {i+1}')
            axes[i].axis('off')
        
        # Set the overall title for the figure
        fig.suptitle(f'Patient: {patient_id}, Slice: {slice_number}')
        fig.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(save_path, f'{patient_id}_slice_{slice_number}_mask_channels.png'))
        plt.close(fig)

    def check_test_results(self):
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
                mask = self.masks.get(pred_patient).data.cpu().numpy().squeeze()
                self._analize_and_plot(patient_id=patient_id, 
                                                    slice_number=slice_number,
                                                    image_niigz_file=image_niigz_file,
                                                    roi_niigz_file=roi_niigz_file,
                                                    mask=mask)
                
                # Save self.masks.get(pred_patient).xy to a .txt file
                output_file = os.path.join(self.output_dir, f"{patient_id}_slice_{slice_number}.txt")
                with open(output_file, "w") as f:
                    xyn_coords = self.masks.get(pred_patient).xyn
                    
                    for contour in xyn_coords:
                        # Format the XY coordinates as a string with numbers separated by spaces
                        xyn_coords_str = " ".join([f"{coord[0]} {coord[1]}" for coord in contour])
                        
                        # Add a "0" before each set of XY coordinates
                        f.write("0 ")
                        f.write(xyn_coords_str)
                        f.write("\n")



def main() -> None:
    ValInstance = localYoloV8(weights_folder="/home/mariopasc/Python/old-dataset/Results-BiggerFCD/weights",
                            yaml_file="./config.yaml",
                            train_csv_path="/home/mariopasc/Python/Projects/BSC_final/epilepsy-displasia-focal-segmentation/csv/BiggerFCD/results.csv",
                            val_output_path="/home/mariopasc/Python/Projects/BSC_final/epilepsy-displasia-focal-segmentation/csv/BiggerFCD/val.csv",
                            dataset_path="/home/mariopasc/Python/old-dataset/BiggerFCD_T2FLAIR_yolov8_dataset",
                            test_results_path="/home/mariopasc/Python/old-dataset/Results-BiggerFCD/test",
                            roi_study_path="/home/mariopasc/Python/Datasets/ds-epilepsy/BiggerFCD_ROIT2FLAIR",
                            study_path="/home/mariopasc/Python/Datasets/ds-epilepsy/BiggerFCD_T2FLAIR",
                            output_dir="/home/mariopasc/Python/old-dataset/Results-BiggerFCD/xyn_coordinates")
    #ValInstance.validateYolov8()
    #ValInstance.val_metrics()
    ValInstance.test_yolov8()
    ValInstance.check_test_results()

if __name__ == "__main__":
    main()