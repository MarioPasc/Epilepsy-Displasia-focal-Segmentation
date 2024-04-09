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

def analyze_dataset(base_path, save_path):
    # Subdirectorios a analizar
    sets = ["train", "val", "test"]
    image_counts = []
    label_counts = []
    total_counts = []
    def check_missing_labels(images_path, labels_path, set_type):
        total_images = os.listdir(images_path)
        total_labels = os.listdir(labels_path)

        # Crear conjuntos de los nombres de archivos sin extensiones para comparar
        image_set = {os.path.splitext(image)[0] for image in total_images}
        label_set = {os.path.splitext(label)[0] for label in total_labels}

        # Crear una lista de imágenes que no tienen etiqueta correspondiente
        images_without_label = [image for image in image_set if image not in label_set]
        return (len(images_without_label))
    # Recorrer cada conjunto y contar archivos
    for set_name in sets:
        image_path = os.path.join(base_path, "images", set_name)
        label_path = os.path.join(base_path, "labels", set_name)
        
        # Contar imágenes y etiquetas
        total = len([name for name in os.listdir(image_path) if name.endswith('.png')])
        num_images = check_missing_labels(images_path=image_path,
                                          labels_path=label_path,
                                           set_type=set_name)

        num_labels = total - num_images

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


