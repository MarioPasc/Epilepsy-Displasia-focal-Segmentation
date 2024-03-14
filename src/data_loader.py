import os
import numpy as np
import cv2 as cv
import nibabel as nib
from sklearn.model_selection import train_test_split


# Esta función convierte un estudio nii.gz a diversas imágenes .nii, asociando como nombre
# de las imágenes idpaciente-slice.nii.
# VER2: Se añade también un parámetro de entrada que es una lista de rodajas que no se deben incluir
def convert_gz_nii(niigz_file, save_path, path_excluir, patient_id):
    # Verificar si el archivo .nii.gz existe
    if not os.path.isfile(niigz_file):
        print("Error: Specified .nii.gz file doesn't exist.")
        return
    # Creamos el vector con las rodajas a exlcuir
    exclude_slices = []
    # Si existe el archivo .txt, rellenamos el vector con las rodajas que hay que excluir
    # Si no existe, se asume que se incluyen todas.
    if os.path.isfile(path_excluir):
        with open(path_excluir, 'r') as archivo:
            for linea in archivo:
                exclude_slices.append(int(linea.strip()))

    # Cargar el archivo .nii.gz
    img = nib.load(niigz_file)
    data = img.get_fdata()
    # Obtener el número de rodajas
    num_slices = data.shape[2]

    # Crear el directorio de destino si no existe
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Iterar sobre todas las rodajas y guardarlas como imágenes .nii
    for i in range(num_slices):
        # Verificar si la rodaja está en la lista de exclusión
        if i in exclude_slices:
            continue  # Saltar esta rodaja

        # Obtener una sola rodaja
        slice_data = data[:, :, i]
        # Crear el nombre de la imagen
        image_name = f"{patient_id}-{i}.nii"
        # Guardar la imagen .nii
        nib.save(nib.Nifti1Image(slice_data, img.affine), os.path.join(save_path, image_name))



# Esta función hace uso de la anterior para convertir un estudio completo a formato .nii
def study_to_nii(study_path, save_path, path_excluir):
    if not os.path.exists(study_path):
        print("Input file does not exist")
        return

    # Por cada paciente dentro del estudio
    for patient_id in os.listdir(study_path):
        # Comprobar si el elemento es una carpeta
        patient_folder = os.path.join(study_path, patient_id)
        if not os.path.isdir(patient_folder):
            print(f"Patient {patient_folder} not found")
            continue

        # Extraemos el nombre de su archivo nii.gz
        niigz_files = [path for path in os.listdir(patient_folder) if path.endswith(".nii.gz")]
        if not niigz_files:
            print(f"No .nii.gz file found for patient {patient_id}")
            continue

        # Asumimos que solo hay un archivo .nii.gz por paciente
        niigz_file = os.path.join(patient_folder, niigz_files[0])
        convert_gz_nii(niigz_file=niigz_file, save_path=save_path, patient_id=patient_id, path_excluir=path_excluir)

    print(f"Estudio {study_path} convertido a nii")


# Esta función toma como entrada una carpeta con imágenes .nii y las guarda en la carpeta de destino
# en el formato especificado.
def convert_nii_image_holdout(input_txt):
    # Verificar si el archivo .txt existe
    if not os.path.isfile(input_txt):
        print("Input txt file does not exist.")
        return

    # Leer el archivo .txt
    with open(input_txt, 'r') as file:
        lines = file.readlines()

    # Obtener las rutas de los folders y archivos
    nii_folder = lines[0].split(':')[1].strip()
    train_folder = lines[1].split(':')[1].strip()
    train_nii_files = lines[2].split(':')[1].strip()
    val_folder = lines[3].split(':')[1].strip()
    val_nii_files = lines[4].split(':')[1].strip()
    test_folder = lines[5].split(':')[1].strip()
    test_nii_files = lines[6].split(':')[1].strip()
    imformat = lines[7].split(':')[1].strip().lower()

    # Función para convertir .nii a imagen
    def convert_to_image(nii_files, folder):
        for nii_file in nii_files:
            # Cargar el archivo .nii
            img = nib.load(os.path.join(nii_folder, nii_file))
            data = img.get_fdata()

            # Verificar si hay valores no válidos en los datos
            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                print(f"Skipping {nii_file}: Invalid values encountered.")
                continue

            # ========== NORMALIZACIÓN PROVISIONAL ==========
            # Normalizar los valores de píxel para que estén entre 0 y 255
            data_normalized = ((data - np.min(data)) / (np.max(data) - np.min(data)) * 255).astype(np.uint8)
            # ========== FIN NORMALIZACIÓN PROVISIONAL ======

            # Obtener el nombre del archivo sin la extensión .nii
            file_name, _ = os.path.splitext(nii_file)
            # Crear el nombre de la imagen con el formato especificado
            image_name = f"{file_name}.{imformat}"
            # Guardar la imagen en el formato especificado
            cv.imwrite(os.path.join(folder, image_name), data_normalized)

    # Obtener los nombres de archivos .nii para train, val y test
    train_nii_files = [file.strip() for file in open(train_nii_files, 'r').readlines()]
    val_nii_files = [file.strip() for file in open(val_nii_files, 'r').readlines()]
    test_nii_files = [file.strip() for file in open(test_nii_files, 'r').readlines()]

    # Convertir .nii a imágenes para train, val y test
    convert_to_image(train_nii_files, train_folder)
    convert_to_image(val_nii_files, val_folder)
    convert_to_image(test_nii_files, test_folder)

    print("Holdout realizado")


# Esta función se encarga del hold out (gracias a dios que existe scikit-learn)
def holdout_nii_images(folder_path, val_percent, test_percent, output_path):
    # Obtener la lista de archivos .nii en la carpeta
    nii_files = [file for file in os.listdir(folder_path) if file.endswith('.nii')]

    # Dividir los nombres de los archivos en train, val y test
    train_files, val_test_files = train_test_split(nii_files, test_size=(val_percent + test_percent), random_state=42)
    val_files, test_files = train_test_split(val_test_files, test_size=test_percent / (val_percent + test_percent),
                                             random_state=42)

    # Escribir los nombres de los archivos en archivos de texto
    def write_to_txt(file_list, txt_path):
        with open(txt_path, 'w') as file:
            for file_name in file_list:
                file.write(file_name + '\n')

    write_to_txt(train_files, os.path.join(output_path, 'train_files.txt'))
    write_to_txt(val_files, os.path.join(output_path, 'val_files.txt'))
    write_to_txt(test_files, os.path.join(output_path, 'test_files.txt'))

    print("Train-Test-Val split realizado")


def extract_roi_contours(input_txt):
    # Verificar si el archivo .txt existe
    if not os.path.isfile(input_txt):
        print("Input txt file does not exist.")
        return

    # Leer el archivo .txt
    with open(input_txt, 'r') as file:
        lines = file.readlines()

    # Obtener las rutas de los folders y archivos
    nii_folder = lines[0].split(':')[1].strip()
    train_folder = lines[1].split(':')[1].strip()
    train_nii_files = lines[2].split(':')[1].strip()
    val_folder = lines[3].split(':')[1].strip()
    val_nii_files = lines[4].split(':')[1].strip()
    test_folder = lines[5].split(':')[1].strip()
    test_nii_files = lines[6].split(':')[1].strip()

    # Función para extraer y guardar contornos
    def save_contours(nii_files, folder):
        for nii_file in nii_files:
            # Cargar el archivo .nii
            img = nib.load(os.path.join(nii_folder, nii_file))
            data = img.get_fdata()

            # Obtener los contornos utilizando cv.findContours
            contours, _ = cv.findContours(data.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            # Obtener las dimensiones de la imagen
            # Cuidado... que las dimensiones de YOLO parecen estar cambiadas
            # así parece funcionar bien, todas las coordenadas están entre 0 y 1
            width = data.shape[1]
            height = data.shape[0]

            # Guardar los contornos en formato YOLO
            output_path = os.path.join(folder, f"{nii_file.strip('.nii')}.txt")
            contours_YOLO_format(contours=contours, height=height, width=width, output_path=output_path)

    # Obtener los nombres de archivos .nii para train, val y test
    train_nii_files = [file.strip() for file in open(train_nii_files, 'r').readlines()]
    val_nii_files = [file.strip() for file in open(val_nii_files, 'r').readlines()]
    test_nii_files = [file.strip() for file in open(test_nii_files, 'r').readlines()]

    # Extraer y guardar contornos para train, val y test
    save_contours(train_nii_files, train_folder)
    save_contours(val_nii_files, val_folder)
    save_contours(test_nii_files, test_folder)

    print("Contornos ROI guardados en formato YOLO")


def contours_YOLO_format(contours, height, width, output_path):
    if len(contours) > 0:
        # YOLOv8 no admite contornos con menos de 2 puntos (tag + x1+y1+x2+y2 = 5)
        if len(contours[0]) >= 5:
            with open(output_path, 'w') as f:
                for i, contorno in enumerate(contours):
                    # Normalizar coordenadas del contorno
                    normalized = contorno.squeeze() / np.array([width, height])
                    # Convertir a formato YOLO
                    str_contour = ' '.join([f"{coord:.6f}" for coord in normalized.flatten()])
                    # Escribir etiqueta 0 y coordenadas
                    f.write(f"0 {str_contour}\n")
        else:
            return
    else:
        return




