#!/bin/bash

# Ruta base donde se encuentran las carpetas images y labels
base_path="/home/mariopasc/Python/Datasets/T2FLAIR-ds-epilepsy"

# Recorre todos los archivos .txt en labels y sus subcarpetas
find "$base_path/labels/" -type f -name "*.txt" | while read txt_file; do
    # Extrae el nombre base sin la extensión .txt
    base_name=$(basename "$txt_file" .txt)

    # Construye la ruta esperada del archivo PNG reemplazando la ruta base de labels por images
    expected_png_file="${txt_file/$base_path\/labels/$base_path\/images}"
    expected_png_file="${expected_png_file%.txt}.png"

    # Comprueba si existe el archivo PNG correspondiente
    if [[ -f "$expected_png_file" ]]; then
        echo "Encontrado: $expected_png_file"
    else
        echo "Falta el archivo PNG correspondiente para: $txt_file"
    fi
done
