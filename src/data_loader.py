import os
import nibabel as nib


def convert_nii_to_nii_slices(niigz_file, save_path, patient_id):
    # Verificar si el archivo .nii.gz existe
    if not os.path.isfile(niigz_file):
        print("Error: El archivo .nii.gz especificado no existe.")
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
        # Obtener una sola rodaja
        slice_data = data[:, :, i]
        # Crear el nombre de la imagen
        image_name = f"{patient_id}-{i}.nii"
        # Guardar la imagen .nii
        nib.save(nib.Nifti1Image(slice_data, img.affine), os.path.join(save_path, image_name))

    print("Process finished with exit code 0")


