#!/bin/bash

# Directorios a verificar
dir1="/home/mariopasc/x2go_shared/t2flair-yolov8-ds-rotr/images/train/"
dir2="/home/mariopasc/x2go_shared/t2flair-yolov8-ds-rotr/labels/train/"
specific_word="rot"
# Extensiones de archivo a considerar en dir2
extensions=("txt") # Añade o elimina según sea necesario

# Recorrer cada archivo en dir1 (ignorando subdirectorios)
for filepath in "${dir1}"/*${specific_word}*; do
  # Comprobar si realmente es un archivo y no un directorio
  if [ -f "$filepath" ]; then
    # Extraer el nombre base del archivo sin la extensión
    filename=$(basename -- "$filepath")
    base_filename="${filename%.*}"

    # Indicar que se está procesando este archivo
    echo "Procesando: $filename"

    # Recorrer cada extensión de archivo para buscar en dir2
    found=0 # Bandera para indicar si se encontró un archivo correspondiente
    for ext in "${extensions[@]}"; do
      # Construir el nombre del archivo a buscar en dir2
      search_filepath="${dir2}/${base_filename}.${ext}"

      # Verificar si el archivo existe en dir2
      if [ -f "$search_filepath" ]; then
        # echo "Correspondencia encontrada: $(basename -- "$search_filepath")"
        found=1
        break # Romper el bucle si se encuentra una correspondencia
      fi
    done

    # Si no se encontró ninguna correspondencia, imprimir un mensaje
    if [ $found -eq 0 ]; then
      echo "No se encontró una correspondencia para: $filename"
    fi
  fi
done
