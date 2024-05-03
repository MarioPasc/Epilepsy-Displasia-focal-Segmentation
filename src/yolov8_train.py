import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO
import numpy as np
import os
import glob
import csv


class SegmentationV8:
    def __init__(self, path_model, yaml_file):
        self.model = self.load_model(path_model=path_model)
        self.yaml_file = yaml_file
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.weights_folder = os.path.join(self.base_dir, "runs/segment/Yolov8-Train/weights")
        self.csv_path = os.path.join(self.base_dir, "runs/segment/Yolov8-Train/results_val.csv")

    def load_model(self, path_model):
        model_path = path_model if os.path.exists(path_model) else "yolov8n-seg.pt"
        model = YOLO(model=model_path, task="segment")
        # model.fuse()
        return model

    def train_yolov8(self):
        # Deshabilitar aumentación de datos: https://github.com/ultralytics/ultralytics/issues/4812
        results_train = self.model.train(data=self.yaml_file, epochs=3, imgsz=640,
                                         save=True, save_period=1,
                                         name="Yolov8-Train", verbose=True,
                                         seed=42, single_cls=True, plots=True,
                                         augment=False, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0, translate=0.0,
                                         scale=0.0, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.0, mosaic=0.0,
                                         close_mosaic=0, mixup=0.0, copy_paste=0.0, auto_augment="", erasing=0.0)

    def validate_yolov8(self):
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

    def val_metrics(self):
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


def main(path_model, yaml_path):
    model = SegmentationV8(path_model=path_model, yaml_file=yaml_path)
    model.train_yolov8()
    model.validate_yolov8()
    model.val_metrics()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train YOLO model.')
    parser.add_argument('--model_path', type=str, default='',
                        help='Path to the .pt model. Leave in blank if training from pre-trained yolo')
    parser.add_argument('--yaml_path', type=str, required=True,
                        help='Path to YAML file')
    args = parser.parse_args()
    main(args.model_path, args.yaml_path)

