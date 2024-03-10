#!/bin/bash

# Definir las carpetas a limpiar
folders=(
    "/home/mariopasc/Python/Datasets/ds-epilepsy/T2flair-study-nii"
    "/home/mariopasc/Python/Datasets/ds-epilepsy/roi-nii"
    "/home/mariopasc/Python/Datasets/t2flair-yolov8-ds/images/train"
    "/home/mariopasc/Python/Datasets/t2flair-yolov8-ds/images/val"
    "/home/mariopasc/Python/Datasets/t2flair-yolov8-ds/images/test"
    "/home/mariopasc/Python/Datasets/t2flair-yolov8-ds/labels/train"
    "/home/mariopasc/Python/Datasets/t2flair-yolov8-ds/labels/val"
    "/home/mariopasc/Python/Datasets/t2flair-yolov8-ds/labels/test"
)

# Eliminar los archivos dentro de cada carpeta
for folder in "${folders[@]}"; do
    rm -r "$folder"/*
done

echo "Archivos eliminados correctamente."
