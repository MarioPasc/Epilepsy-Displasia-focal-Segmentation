import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import nibabel as nib
from typing import List, Tuple, Dict
import cv2
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import glob
import matplotlib
matplotlib.use('Agg')

class DataExplore:

    def __init__(self, study_path: str, roi_path: str, dataset_path: str, save_path: str) -> None:
        self.study_path = study_path
        self.roi_path = roi_path
        self.dataset_path = dataset_path
        self.save_path = save_path
        self.valid_roi_slices: Dict[str, List[int]] = {}
        self.__find_valid_roi_slices__()

    # Find valid slices for each patient. We consider a slice as valir
    # if its corresponding roi's contour has at least 5 points
    def __process_roi_file__(self, roi_file: str) -> Tuple[str, List[int]]:
        valid_slices = []
        roi_path = os.path.join(self.roi_path, roi_file)
        roi_img = nib.load(roi_path)
        roi_data = roi_img.get_fdata()

        for i in range(roi_data.shape[2]):
            slice_data = roi_data[:, :, i]
            slice_data = np.array(slice_data, dtype=np.uint8)
            contours, _ = cv2.findContours(slice_data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours and math.ceil(len(contours[0]) / 2) > 3:
                valid_slices.append(i)

        return (roi_file, valid_slices)

    # Apply the function process_roi_file concurrently
    def __find_valid_roi_slices__(self) -> None:
        with ThreadPoolExecutor() as executor:
            futures = []

            for roi_file in os.listdir(self.roi_path):
                if roi_file.endswith('.nii.gz'):
                    futures.append(executor.submit(self.__process_roi_file__, roi_file))

            for future in as_completed(futures):
                patient_id, valid_slices = future.result()
                if valid_slices:
                    self.valid_roi_slices[patient_id] = valid_slices

    def dataset_integrity(self):
        exclude = list(range(0, 121)) + list(range(200, 257))
        total_images = []
        images_found = []
        total_labels = []
        labels_found = []
        for file_name in self.valid_roi_slices.keys():
            id_patient, augmentation_type = file_name.split('_')[0], file_name.split('_')[-2]
            if augmentation_type in ["gamma", "brightness", "flip", "shift"]:
                output_format_image = f"{id_patient}_slice-{{slice}}_{augmentation_type}.png"
                output_format_label = f"{id_patient}_slice-{{slice}}_{augmentation_type}.txt"
            else:
                output_format_image = f"{id_patient}_slice-{{slice}}.png"
                output_format_label = f"{id_patient}_slice-{{slice}}.txt"
            for slice in self.valid_roi_slices.get(file_name):
                if augmentation_type not in ["gamma", "brightness", "flip", "shift"] and slice in exclude:
                    continue
                output_file_image = output_format_image.format(slice=slice)
                output_file_label = output_format_label.format(slice=slice)
                file_found = False  # Control variable
                for set_folder in ["train", "val", "test"]:
                    image_path = os.path.join(self.study_path, "images", set_folder, output_file_image)
                    label_path = os.path.join(self.study_path, "labels", set_folder, output_file_label)
                    
                    total_labels.append(label_path)
                    total_images.append(image_path)

                    if os.path.exists(image_path) and os.path.exists(label_path):
                        file_found = True
                        total_labels.append(label_path)
                        images_found.append(image_path)
                        # File found in this set, no need to check further
                        break  # Exit the for loop for set_folder
        return total_images, images_found, total_labels, labels_found
    
    def plot_this(self, total_images, images_found, total_labels, labels_found):
        categories = ['Total Images', 'Images Found', 'Total Labels', 'Labels Found']
        values = [len(total_images), len(images_found), len(total_labels), len(labels_found)]
        
        plt.figure(figsize=(10, 6))
        plt.bar(categories, values, color=['blue', 'green', 'red', 'orange'])
        plt.title('Dataset Integrity Check')
        plt.ylabel('Count')
        for i, v in enumerate(values):
            plt.text(i, v + 5, str(v), ha='center')
        
        diff_images = 100 * (len(total_images) - len(images_found)) / len(total_images)
        diff_labels = 100 * (len(total_labels) - len(labels_found)) / len(total_labels)
        print(f"Diferencia en % (Imágenes): {diff_images}%")
        print(f"Diferencia en % (Etiquetas): {diff_labels}%")
        
        plt.savefig(os.path.join(self.save_path, "dataset_integrity.png"))
        plt.close()

def analyze_dataset(base_path, save_path):
    # Subdirectorios a analizar
    sets = ["train", "val", "test"]
    image_counts = []
    label_counts = []
    total_counts = []

    # Recorrer cada conjunto y contar archivos
    for set_name in sets:
        image_path = os.path.join(base_path, "images", set_name)
        label_path = os.path.join(base_path, "labels", set_name)
        
        # Contar imágenes y etiquetas
        total = len([name for name in os.listdir(image_path) if name.endswith('.png')])
        num_labels = len([name for name in os.listdir(label_path) if name.endswith('.txt')])
        
        num_images = total - num_labels

        total_counts.append(total)
        image_counts.append(num_images)
        label_counts.append(num_labels)

    x = np.arange(len(sets))  # Posiciones de las etiquetas en el eje x
    width = 0.25 # Ancho de las barras

    if not total_counts or len(total_counts) != len(sets):
        print("Error: total_counts is empty or does not match the number of sets.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    rects2 = ax.bar(x - width, image_counts, width, label='FCD=0', color='skyblue')
    rects3 = ax.bar(x, label_counts, width, label='FCD=1', color='orange')
    rects1 = ax.bar(x + width, total_counts, width, label='Total Images', color='lightgreen')

    # Etiquetas y títulos
    ax.set_ylabel('Count')
    ax.set_title('Proportion of Images with and without Label by Set')
    ax.set_xticks(x)
    ax.set_xticklabels(sets)
    ax.legend()

    # Función para añadir etiquetas de conteo sobre las barras
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 puntos verticales de offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    # Añadir etiquetas de conteo
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    fig.tight_layout()

    # Verificar y crear el directorio de guardado si no existe
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Guardar el gráfico
    plt.savefig(os.path.join(save_path, "dataset_distribution.png"))

    # Cerrar la figura para liberar memoria
    plt.close()




def main():
    results_path = "/home/mariopasc/Python/Projects/BSC_final/epilepsy-displasia-focal-segmentation/images"
    dataExplorer = DataExplore(study_path="/home/mariopasc/Python/Datasets/ds-epilepsy/T2FLAIR",
                               roi_path="/home/mariopasc/Python/Datasets/ds-epilepsy/ROI_T2",
                               dataset_path="/home/mariopasc/Python/Datasets/T2FLAIR-ds-epilepsy",
                               save_path=results_path)
    analyze_dataset("/home/mariopasc/Python/Datasets/T2FLAIR-ds-epilepsy", 
                    results_path)

if __name__ == "__main__":
    main()
