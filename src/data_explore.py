import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def analyze_dataset(folder_paths, label_file_path, save_path):
    # Verificar y crear el directorio de guardado si no existe
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Leer los nombres de los archivos con label positivo
    with open(label_file_path, 'r') as file:
        labeled_images = set(file.read().splitlines())
    labeled_images = [name.rstrip(".nii") for name in labeled_images]
    labeled_counts = []
    total_images_counts = []

    for folder_path in folder_paths:
        labeled_count = 0
        total_images = 0
        # Preparar una lista de todos los archivos antes de iniciar el bucle para poder usar tqdm
        all_files = []
        for root, _, files in os.walk(folder_path):
            for name in files:
                if name.endswith('.png'):
                    all_files.append((root, name))
        # Utilizar tqdm aquí para mostrar la barra de progreso
        for root, name in tqdm(all_files, desc=f"Procesando {folder_path}"):
            total_images += 1
            image_base_name = name.split('.png')[0]
            # Considerar tanto imágenes directamente etiquetadas como aquellas aumentadas
            if any(image_base_name.startswith(labeled_image) for labeled_image in labeled_images):
                labeled_count += 1

        labeled_counts.append(labeled_count)
        total_images_counts.append(total_images)

    labeled_percentages = np.array(labeled_counts) / np.array(total_images_counts) * 100
    non_labeled_counts = np.array(total_images_counts) - np.array(labeled_counts)
    non_labeled_percentages = np.array(non_labeled_counts) / np.array(total_images_counts) * 100

    labels = ['Train', 'Validation', 'Test']
    x = np.arange(len(labels))
    width = 0.20  # Ajuste el ancho para acomodar una barra adicional

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, total_images_counts, width, label='Total Images', color='skyblue')
    rects2 = ax.bar(x, labeled_counts, width, label='Images with Label', color='orange')
    rects3 = ax.bar(x + width, non_labeled_counts, width, label='Images without Label', color='lightgreen')

    ax.set_ylabel('Images Count')
    ax.set_title('Proportion of Images with and without Label by Set')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Función para añadir etiquetas de porcentaje sobre las barras
    def autolabel(rects, percentages):
        for rect, pct in zip(rects, percentages):
            height = rect.get_height()
            ax.annotate(f'{pct:.2f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 puntos verticales de offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    # Aplicar la función autolabel para añadir porcentajes de imágenes con y sin label
    autolabel(rects2, labeled_percentages)
    autolabel(rects3, non_labeled_percentages)

    fig.tight_layout()

    # Guardar el gráfico en el directorio especificado
    save_file_path = os.path.join(save_path, "dataset_labels_distribution.png")
    plt.savefig(save_file_path)
    print(f"Gráfico guardado en: {save_file_path}")

    plt.close()


def main():
    analyze_dataset(folder_paths=[
        "/home/mariopasc/Python/Datasets/t2flair-yolov8-ds/images/train",
        "/home/mariopasc/Python/Datasets/t2flair-yolov8-ds/images/val",
        "/home/mariopasc/Python/Datasets/t2flair-yolov8-ds/images/test"
    ], label_file_path="/home/mariopasc/Python/Projects/BSC_final/epilepsy-displasia-focal-segmentation/info-files"
                       "/t2flair-study/roi-slices.txt",
        save_path="/home/mariopasc/Python/Projects/BSC_final/epilepsy-displasia-focal-segmentation/info-files/t2flair"
                  "-study/images")


if __name__ == "__main__":
    main()
