#!/bin/bash

output_dir=~/Python/Projects/BSC_final/epilepsy_segment/bash-scripts/pacientes-test

mkdir -p "$output_dir"
output_file="$output_dir/pacientes-test.txt"

for sub_dir in ~/Python/Datasets/ds004199-download/sub-*; do
    # Nombre del subdirectorio sin la ruta
    sub_dirname=$(basename "$sub_dir")

    roi_file=$(find "$sub_dir/anat" -name "*roi*" -type f)

    # Verificar si no se encontró ningún archivo roi en la carpeta 'anat' del paciente
    if [ -z "$roi_file" ]; then
        echo "$sub_dirname" >> "$output_file"
    fi
done
