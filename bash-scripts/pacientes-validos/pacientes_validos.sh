#!/bin/bash

# Ruta al fichero de entrada con los nombres de las carpetas
input_file="/home/mario/VSCode/Projects/Epilepsy-Displasia-focal-Segmentation/bash-scripts/pacientes-validos"

# Directorio de origen
source_dir="/home/mario/VSCode/Dataset/epilepsy"

# Directorio de destino
destination_dir="/home/mario/VSCode/Dataset/ds-epilepsy"

# Verifica si el directorio de destino existe, si no, créalo
if [ ! -d "$destination_dir" ]; then
    mkdir -p "$destination_dir"
fi

# Lee el fichero de entrada línea por línea
while IFS= read -r folder_name; do
    # Comprueba si el directorio existe en el directorio de origen
    if [ -d "$source_dir/$folder_name" ]; then
        # Mueve el directorio al directorio de destino
        mv "$source_dir/$folder_name" "$destination_dir/"
        echo "Carpeta '$folder_name' movida correctamente."
    else
        echo "Advertencia: Carpeta '$folder_name' no encontrada en el directorio de origen."
    fi
done < "$input_file"

echo "Proceso completado."
