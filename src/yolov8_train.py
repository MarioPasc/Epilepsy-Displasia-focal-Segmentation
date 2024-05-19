import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO
import numpy as np
import os
import glob
import csv
import yaml
import nibabel as nib
import cv2
import torch

class SegmentationV8:
    def __init__(self, path_model, yaml_file):
        torch.cuda.empty_cache()
        self.model = self.load_model(path_model=path_model)
        self.yaml_file = yaml_file
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.weights_folder = os.path.join(self.base_dir, "runs/segment/Yolov8-Train/weights")
        self.csv_path = os.path.join(self.base_dir, "runs/segment/Yolov8-Train/results_val.csv")
        self.test_results = os.path.join(self.base_dir, "runs/segment/Yolov8-Train/test_results")
        self.stats_df = None

    def load_model(self, path_model):
        model_path = path_model if os.path.exists(path_model) else "yolov8n-seg.pt"
        model = YOLO(model=model_path, task="segment")
        # model.fuse()
        return model

    def train(self):
        # Deshabilitar aumentación de datos: https://github.com/ultralytics/ultralytics/issues/4812
        results_train = self.model.train(data=self.yaml_file, epochs=100, imgsz=640,
                                         save=True, save_period=1,
                                         name="Yolov8-Train", verbose=True,
                                         seed=42, single_cls=True, plots=True,
                                         augment=False, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0, translate=0.0,
                                         scale=0.0, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.0, mosaic=0.0,
                                         close_mosaic=0, mixup=0.0, copy_paste=0.0, auto_augment="", erasing=0.0)

    def val(self):
        weights_files = sorted([file for file in glob.glob(os.path.join(self.weights_folder, '*.pt'))
                                if 'last.pt' not in file])

        # Verificar si 'last.pt' existe y agregarlo al final de la lista
        last_pt_path = os.path.join(self.weights_folder, 'last.pt')
        if os.path.exists(last_pt_path):
            weights_files.append(last_pt_path)

        epoch_counter = 1 
        with open(self.csv_path, 'w', newline='') as csvfile:
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
        self._val_metrics()

    def _val_metrics(self):
        """
        This function plots the training metrics versus validation metrics for each epoch
        using data from 'results.csv' and 'results_val.csv'.
        """
        # Load training and validation metrics
        train_metrics_df = pd.read_csv(os.path.join(self.base_dir, "runs/segment/Yolov8-Train/results.csv"))
        train_metrics_df.columns = train_metrics_df.columns.str.strip()  
        val_metrics_df = pd.read_csv(self.csv_path)

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
            save_path = os.path.join(self.base_dir, "runs/segment/Yolov8-Train/metrics")
            print(save_path)
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(f"{save_path}/{metric_name.replace('metrics/', '')}_train_vs_val_per_epoch.png")
            plt.close()

        # Define the metrics you want to plot
        metric_names = ['precision(M)', 'recall(M)', 'mAP50(M)', 'mAP50-95(M)',
                        'precision(B)', 'recall(B)', 'mAP50(B)', 'mAP50-95(B)']

        for metric_name in metric_names:
            plot_metric(train_metrics_df, val_metrics_df, metric_name)

    def test(self) -> None:
        with open("./config.yaml", "r") as file:
            config = yaml.safe_load(file)
        os.makedirs(os.path.join(self.base_dir, "runs/segment/Yolov8-Train/test_results"), exist_ok=True)

        dataset_path = config["path"]
        self.test_data_path = os.path.join(dataset_path, "images", "test")

        predict_files = [file for file in os.listdir(self.test_data_path) if file.endswith(".png")]
        if not predict_files:
            raise ValueError(f"No test files found in {self.test_data_path}")

        predict_files = [os.path.join(self.test_data_path, file) for file in predict_files]
        best_pt = os.path.join(self.weights_folder, "best.pt")
        model = YOLO(best_pt)

        self.masks = {}
        self.stats_df = pd.DataFrame(columns=["patient_id", "slice", "exist", "dice_score"])

        for file in predict_files:
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
                im_pred.save(filename=os.path.join(self.test_results, f"{os.path.basename(file)}_predict.png"))
            print(os.path.basename(file))
        self.stats_df.to_csv(os.path.join(self.test_results, "test_stats.csv"), index=False)
        self._evaluate_test()

    def _dice_score(self, mask1, mask2) -> float:
        intersection = np.sum((mask1 > 0) & (mask2 > 0))
        total = np.sum(mask1 > 0) + np.sum(mask2 > 0)
        if total == 0:
            return 1.0  # Si ambas máscaras están vacías, el score es perfecto
        return 2 * intersection / total

    def _evaluate_test(self) -> None:
        roi_masks = os.path.join(self.base_dir, "test_roi_masks")

        for patient_prediction in self.masks.keys():
            patient_prediction = patient_prediction.strip(".png")
            patient = patient_prediction.split("_")[0]
            slice = patient_prediction.split("-")[-1]

            niigz = glob.glob(os.path.join(roi_masks, f"{patient}*.nii.gz"))
            if len(niigz) > 0:
                patient = nib.load(niigz[0]).get_fdata()
                ground_truth = np.uint8(patient)[:, :, int(slice)]
                prediction = self.masks.get(patient_prediction)
                image_slice = cv2.imread(os.path.join(self.test_data_path, f"{patient_prediction}.png"), cv2.IMREAD_GRAYSCALE)
                self._analyze_and_plot_test(masks=prediction, roi_slice=ground_truth, image_slice=image_slice, 
                                            patient_id=patient, slice_number=slice, iterations=1, kernel_size=3)
                
    def _analyze_and_plot_test(self, masks, roi_slice, image_slice, patient_id, slice_number, iterations, kernel_size) -> None:
        merged_tensor = np.zeros(shape=(masks.shape[1], masks.shape[2]))
        for i in range(masks.shape[0]):
            merged_tensor += masks[i]
        merged_tensor = np.clip(merged_tensor, 0, 1)
        roi_slice = cv2.resize(roi_slice, (merged_tensor.shape[1], merged_tensor.shape[0]), interpolation=cv2.INTER_AREA)
        # Apply erosion to the ground truth mask 
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        eroded_roi_slice = cv2.erode(roi_slice, kernel, iterations=iterations)
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
        
        ax2.imshow(image_slice, cmap='gray')
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
        plt.savefig(os.path.join(self.test_results, f'{patient_id}_slice_{slice_number}_comparison.png'))
        plt.close(fig)            





def main(path_model, yaml_path) -> None:
    model = SegmentationV8(path_model=path_model, yaml_file=yaml_path)
    model.train()
    model.val()
    model.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train YOLO model.')
    parser.add_argument('--model_path', type=str, default='',
                        help='Path to the .pt model. Leave in blank if training from pre-trained yolo')
    parser.add_argument('--yaml_path', type=str, required=True,
                        help='Path to YAML file')
    args = parser.parse_args()
    main(args.model_path, args.yaml_path)

