import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


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
        num_images = len([name for name in os.listdir(image_path) if name.endswith('.png')])
        num_labels = len([name for name in os.listdir(label_path) if name.endswith('.txt')])
        
        total = num_images + num_labels
        total_counts.append(total)
        
        image_counts.append(num_images)
        label_counts.append(num_labels)

    x = np.arange(len(sets))  # Posiciones de las etiquetas en el eje x
    width = 0.25 # Ancho de las barras

    if not total_counts or len(total_counts) != len(sets):
        print("Error: total_counts is empty or does not match the number of sets.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    rects2 = ax.bar(x - width, image_counts, width, label='Images', color='skyblue')
    rects3 = ax.bar(x, label_counts, width, label='Labels', color='orange')
    rects1 = ax.bar(x + width, total_counts, width, label='Images + Labels', color='lightgreen')

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
    path = "/home/mariopasc/Python/Datasets/T2FLAIR-ds-epilepsy"
    analyze_dataset(base_path=path,
                    save_path="/home/mariopasc/Python/Projects/BSC_final/epilepsy-displasia-focal-segmentation/images")


if __name__ == "__main__":
    main()
