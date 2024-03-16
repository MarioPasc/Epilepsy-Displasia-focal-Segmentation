import os
import matplotlib.pyplot as plt
import numpy as np


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
        total_images = len(os.listdir(folder_path))
        images = [name.rstrip(".png") for name in os.listdir(folder_path)]
        print(images)
        print(labeled_images)
        for image_name in images:
            if image_name in labeled_images:
                labeled_count += 1

        labeled_counts.append(labeled_count)
        total_images_counts.append(total_images)

    labeled_percentages = np.array(labeled_counts) / np.array(total_images_counts) * 100

    labels = ['Train', 'Validation', 'Test']
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width / 2, total_images_counts, width, label='Total de Imágenes', color='skyblue')
    ax.bar(x + width / 2, labeled_counts, width, label='Imágenes con Label', color='orange')

    ax.set_ylabel('Número de Imágenes')
    ax.set_title('Proporción de Imágenes con y sin Label por Conjunto')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    for i in range(len(labels)):
        plt.text(i, total_images_counts[i], f"{labeled_percentages[i]:.2f}%", ha='center', color='black', fontsize=12)

    fig.tight_layout()

    # Guardar el gráfico en el directorio especificado
    save_file_path = os.path.join(save_path, "dataset_labels_distribution.png")
    plt.savefig(save_file_path)
    print(f"Gráfico guardado en: {save_file_path}")

    # Limpiar para evitar sobreposiciones en futuras llamadas a plot
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
