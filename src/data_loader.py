import os
import numpy as np
import cv2 as cv
import nibabel as nib


def convert_gz_nii(niigz_file, save_path, patient_id, exclude_slices):
    # Verificar si el archivo .nii.gz existe
    if not os.path.isfile(niigz_file):
        print("Error: Specified .nii.gz file doesn't exist.")
        return

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

    print("Process finished with exit code 0")


def study_to_nii(study_path, save_path, exclude_slices):
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
        convert_gz_nii(niigz_file=niigz_file, save_path=save_path, patient_id=patient_id, exclude_slices=exclude_slices)


def convert_nii_to_image(input_folder, output_folder, imformat):
    # Verificar si el directorio de entrada existe
    if not os.path.isdir(input_folder):
        print("Input folder does not exist.")
        return
    # Crear el directorio de salida si no existe
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Obtener la lista de archivos .nii en el directorio de entrada
    nii_files = [file for file in os.listdir(input_folder) if file.endswith('.nii')]

    for nii_file in nii_files:
        # Cargar el archivo .nii
        img = nib.load(os.path.join(input_folder, nii_file))
        data = img.get_fdata()

        # Verificar si hay valores no válidos en los datos
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            print(f"Skipping {nii_file}: Invalid values encountered.")
            continue

        # Verificar si el rango de los datos es válido
        min_val = np.min(data)
        max_val = np.max(data)
        if min_val == max_val:
            print(f"Skipping {nii_file}: Invalid data range.")
            continue

        # Normalizar los valores de píxel para que estén entre 0 y 255
        data_normalized = ((data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        # Obtener el nombre del archivo sin la extensión .nii
        file_name, _ = os.path.splitext(nii_file)
        # Crear el nombre de la imagen con el formato especificado
        image_name = f"{file_name}.{imformat.lower()}"
        # Guardar la imagen en el formato especificado
        cv.imwrite(os.path.join(output_folder, image_name), data_normalized)

    print("Process finished with exit code 0.")




