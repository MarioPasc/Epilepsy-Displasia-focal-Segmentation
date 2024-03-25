#!/bin/bash

# Definir las carpetas a limpiar
folders=(
    "/home/mario/VSCode/Dataset/ds-epilepsy/T2FLAIR",
    "/home/mario/VSCode/Dataset/ds-epilepsy/ROI",
    "/home/mario/VSCode/Dataset/ds-epilepsy/T1WEIGHTED"
)

# Eliminar los archivos dentro de cada carpeta
for folder in "${folders[@]}"; do
    rm -r "$folder"/*
done

echo "Archivos eliminados correctamente."
