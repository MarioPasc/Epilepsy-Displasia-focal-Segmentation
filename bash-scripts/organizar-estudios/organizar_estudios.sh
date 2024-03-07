#!/bin/bash

# Directorio de origen
source_dir="/home/mariopasc/Python/Datasets/ds-epilepsy"

# Directorio de destino para los ficheros T1w
destination_t1w_dir="/home/mariopasc/Python/Datasets/ds-epilepsy/T1w-study"

# Directorio de destino para los ficheros T2sel_FLAIR
destination_t2flair_dir="/home/mariopasc/Python/Datasets/ds-epilepsy/T2flair-study"

# Directorio de destino para los ficheros roi
destination_roi_dir="/home/mariopasc/Python/Datasets/ds-epilepsy/roi"

# Función para crear directorio si no existe
create_directory() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
    fi
}

# Iterar sobre las carpetas en el directorio de origen
for folder in "$source_dir"/*/; do
    folder_name=$(basename "$folder")
    # Comprobar si existe la carpeta 'anat' dentro de la carpeta actual
    if [ -d "$folder/anat" ]; then
        # Crear los directorios de destino si no existen
        create_directory "$destination_t1w_dir/$folder_name"
        create_directory "$destination_t2flair_dir/$folder_name"
        create_directory "$destination_roi_dir/$folder_name"
        # El orden importa, ya que los ficheros roi tienen los caracteres T2set_FLAIR en su nombre

        # Mover los ficheros T1w a su directorio de destino correspondiente
        find "$folder/anat" -type f -name "*T1w*" -exec mv -t "$destination_t1w_dir/$folder_name" {} +
        # Mover los ficheros roi a su directorio de destino correspondiente
        find "$folder/anat" -type f -name "*roi*" -exec mv -t "$destination_roi_dir/$folder_name" {} +
        # Mover los ficheros T2sel_FLAIR a su directorio de destino correspondiente
        find "$folder/anat" -type f -name "*T2sel_FLAIR*" -exec mv -t "$destination_t2flair_dir/$folder_name" {} +
        echo "Ficheros movidos para la carpeta $folder_name."
    else
        echo "Advertencia: Carpeta 'anat' no encontrada en $folder_name."
    fi
done

echo "Proceso completado."
