import os
import cv2 as cv
import nibabel as nib
import numpy as np
import random
import shutil
import math


def save_slices_with_roi(input_dir, output_txt):
    # Verificar si el directorio de entrada existe
    if not os.path.isdir(input_dir):
        print("Input directory does not exist.")
        return

    # Crear una lista para almacenar los nombres de los archivos con ROI
    files_with_roi = []

    # Obtener la lista de carpetas de pacientes
    patient_folders = [folder for folder in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, folder))]

    for patient_folder in patient_folders:
        # Directorio de entrada para el paciente actual
        patient_input_path = os.path.join(input_dir, patient_folder)

        # Obtener la lista de archivos .nii.gz en el directorio del paciente
        nii_files = [file for file in os.listdir(patient_input_path) if file.endswith('.nii.gz')]
        for nii_file in nii_files:
            # Cargar el archivo .nii.gz
            img = nib.load(os.path.join(patient_input_path, nii_file))
            data = np.uint8(img.get_fdata())

            # Verificar si hay ROI en alguna rodaja
            for i in range(data.shape[2]):
                slice_data = data[:, :, i]
                contours, _ = cv.findContours(slice_data, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                # Contours are given as tuples. If there are at least 5 <x> <y> points (3 <x, y> tuples), consider the contour
                if len(contours) > 0:
                    if math.ceil(len(contours[0])/2) >= 3:
                        files_with_roi.append(f"{patient_folder}-{i}.nii\n")

    # Escribir la lista de archivos con ROI en el archivo de salida
    with open(output_txt, 'w') as file:
        file.writelines(files_with_roi)


def check_augmented_data(input_folder, patient_slice, augmentation_method):
    """
    Comprueba si ya existen archivos con el nombre aumentado en el directorio especificado.
    Si es así, incrementa el número de versión para el nuevo archivo aumentado.
    """
    # Formato base del nombre para archivos aumentados
    base_augmented_image_name = f"{patient_slice}-{augmentation_method.lower()}"

    # Lista todos los archivos en el directorio que coinciden con el patrón base
    matching_files = [f for f in os.listdir(input_folder) if
                      f.startswith(base_augmented_image_name) and f.endswith('.nii')]

    # Si no hay archivos coincidentes, simplemente añade '-0' antes de la extensión
    if not matching_files:
        return f"{base_augmented_image_name}-0.nii"

    # Extrae los números de versión de los archivos existentes y encuentra el más alto
    versions = [int(f.split('-')[-1].split('.')[0]) for f in matching_files]
    highest_version = max(versions)

    # Crea el nombre del nuevo archivo incrementando el número de versión más alto en 1
    new_version = highest_version + 1
    new_augmented_image_name = f"{base_augmented_image_name}-{new_version}.nii"

    return new_augmented_image_name


def copy_augmented_image_with_roi(input_folder_roi, original_image_name, augmentation_method):
    patient_slice = original_image_name.rstrip(".nii")
    augmented_image_name = check_augmented_data(input_folder=input_folder_roi,
                                                patient_slice=patient_slice,
                                                augmentation_method=augmentation_method)
    source_path = os.path.join(input_folder_roi, original_image_name)
    output_folder = os.path.join(input_folder_roi, augmented_image_name)

    shutil.copy2(source_path, output_folder)
    print(f"Augmented image with ROI copied to: {output_folder}")


def gamma_correction_augmentation(image_data):
    gamma_value = random.uniform(.8, 1.2)
    # Aplicar la corrección gamma a los valores de píxeles de la imagen
    corrected_image = np.power(image_data, gamma_value)
    corrected_image = np.clip(corrected_image, a_min=np.min(image_data), a_max=np.max(image_data))
    return corrected_image


def brightness_augmentation(image_data):
    brightness_value = random.uniform(-8, 8)
    # Aplicar el cambio de brillo a los valores de píxeles de la imagen
    modified_image = image_data + brightness_value
    modified_image = np.clip(modified_image, a_min=np.min(image_data), a_max=np.max(image_data))
    return modified_image


def translation_augmentation(image_data, roi_folder, image_name):
    # Generar valores aleatorios para la translación en x e y entre -5 y 5 píxeles
    tx = random.randint(-5, 5)
    ty = random.randint(-5, 5)
    # Definir la matriz de transformación para la translación
    M = np.float32([[1, 0, tx], [0, 1, ty]])

    # Aplicar la translación a la imagen utilizando warpAffine de OpenCV
    translated_image = cv.warpAffine(image_data, M, (image_data.shape[1], image_data.shape[0]))

    # Encontrar el ROI asociado a la imagen
    roi_image = nib.load(os.path.join(roi_folder, image_name)).get_fdata()
    translated_roi = cv.warpAffine(roi_image, M, dsize=(roi_image.shape[1], roi_image.shape[0]))
    return translated_image, translated_roi


def rotation_augmentation(image_data, roi_folder, image_name):
    # Generar un ángulo aleatorio entre -6 y 6 grados
    angle = random.uniform(-6, 6)
    # Obtener el centro de la imagen
    center = (image_data.shape[1] / 2, image_data.shape[0] / 2)
    # Definir la matriz de rotación
    M = cv.getRotationMatrix2D(center, angle, 1.0)

    # Aplicar la rotación a la imagen utilizando warpAffine de OpenCV a la imagen
    rotated_image = cv.warpAffine(image_data, M, dsize=(image_data.shape[1], image_data.shape[0]))

    # Encontrar el ROI asociado a la imagen
    roi_image = nib.load(os.path.join(roi_folder, image_name)).get_fdata()
    rotated_roi = cv.warpAffine(roi_image, M, dsize=(roi_image.shape[1], roi_image.shape[0]))

    return rotated_image, rotated_roi


def check_output_roi_contour(augmented_roi, augmented_data, roi_folder, augmented_image_name, input_folder, img):
    contours_original, _ = cv.findContours(np.uint8(augmented_roi), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if len(augmented_roi) > 0:  # There are contours in the ROI
        if math.ceil(len(augmented_roi[0]) / 2) >= 3:  # There are at least 3 tuples <x, y> in the ROI
            augmented_roi_path = os.path.join(roi_folder, augmented_image_name)
            nib.save(nib.Nifti1Image(augmented_roi, img.affine), augmented_roi_path)
            augmented_image_path = os.path.join(input_folder, augmented_image_name)
            nib.save(nib.Nifti1Image(augmented_data, img.affine), augmented_image_path)
            print(f"Translation augmentation applied to {augmented_image_name}")


def apply_augmentation(num_images, input_folder, roi_folder, roi_txt_file, exclusion_file, augmentation_method):
    # Verificar si el directorio de entrada existe
    if not os.path.isdir(input_folder):
        print("Input folder does not exist.")
        return

    # Verificar si el archivo de texto con las imágenes con ROI existe
    if not os.path.isfile(roi_txt_file):
        print("ROI text file does not exist.")
        return

    # Verificar si el archivo de exclusión de cortes existe
    if not os.path.isfile(exclusion_file):
        print("Exclusion file does not exist.")
        return

    # Leer el archivo de texto con las imágenes con ROI
    with open(roi_txt_file, 'r') as file:
        roi_images = file.readlines()

    # Obtener una lista de imágenes con ROI
    roi_images = [image.strip() for image in roi_images]

    # Verificar si hay suficientes imágenes con ROI
    if len(roi_images) < num_images:
        print("Number of images specified exceeds the number of images with ROI.")
        return

    # Leer el archivo de exclusión de cortes
    with open(exclusion_file, 'r') as file:
        exclude_slices = set(int(line.strip()) for line in file)

    selected_images = []
    while len(selected_images) < num_images:
        # Seleccionar aleatoriamente una imagen con ROI
        random_image = random.choice(roi_images)
        # Obtener el número de rodaja de la imagen seleccionada
        slice_number = int(random_image.split('-')[-1].split('.')[0])
        # Verificar si el número de rodaja está en el conjunto de cortes excluidos
        if slice_number in exclude_slices:
            continue  # Repetir el proceso si el corte está excluido

        else:
            selected_images.append(random_image)

    # Aplicar el método de aumento de datos a cada imagen seleccionada
    for image_name in selected_images:
        # Obtener la ruta de la imagen .nii
        image_path = os.path.join(input_folder, f"{image_name}")
        patient_slice = image_name.rsplit(".nii")[0]
        # Cargar la imagen .nii
        img = nib.load(image_path)
        data = img.get_fdata()

        augmented_image_name = check_augmented_data(input_folder=roi_folder,
                                                    patient_slice=patient_slice,
                                                    augmentation_method=augmentation_method)

        if augmentation_method == "GAMMA":
            # Aplicar el método de aumento de datos
            augmented_data = gamma_correction_augmentation(data)

            # Duplicar el ROI
            copy_augmented_image_with_roi(input_folder_roi=roi_folder,
                                          original_image_name=image_name,
                                          augmentation_method=augmentation_method)
            print(f"Gamma augmentation applied to {augmented_image_name}")
            augmented_image_path = os.path.join(input_folder, augmented_image_name)
            nib.save(nib.Nifti1Image(augmented_data, img.affine), augmented_image_path)

        if augmentation_method == "BRIGHTNESS":
            # Aplicar el método de aumento de datos
            augmented_data = brightness_augmentation(data)

            # Duplicar el ROI
            copy_augmented_image_with_roi(input_folder_roi=roi_folder,
                                          original_image_name=image_name,
                                          augmentation_method=augmentation_method)
            print(f"Brightness augmentation applied to {augmented_image_name}")
            augmented_image_path = os.path.join(input_folder, augmented_image_name)
            nib.save(nib.Nifti1Image(augmented_data, img.affine), augmented_image_path)

        if augmentation_method == "TRANSLATION":
            # Aplicar el método de aumento de datos
            augmented_data, augmented_roi = translation_augmentation(image_data=data,
                                                                     roi_folder=roi_folder,
                                                                     image_name=image_name)
            check_output_roi_contour(augmented_roi=augmented_roi,
                                     augmented_data=augmented_data,
                                     roi_folder=roi_folder,
                                     augmented_image_name=augmented_image_name,
                                     input_folder=input_folder,
                                     img=img)

        if augmentation_method == "ROTATION":
            # Aplicar el método de aumento de datos
            augmented_data, augmented_roi = rotation_augmentation(image_data=data,
                                                                  roi_folder=roi_folder,
                                                                  image_name=image_name)
            check_output_roi_contour(augmented_roi=augmented_roi,
                                     augmented_data=augmented_data,
                                     roi_folder=roi_folder,
                                     augmented_image_name=augmented_image_name,
                                     input_folder=input_folder,
                                     img=img)



