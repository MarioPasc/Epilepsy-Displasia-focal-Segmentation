# Epilepsy Lesion Segmentation using YOLOv8
*(work in progress)*


This project focuses on employing the YOLOv8 neural network architecture for segmenting epilepsy lesions, specifically focal cortical dysplasia type II, in MRI images. The dataset used for training and evaluation is the "An open presurgery MRI dataset of people with epilepsy and focal cortical dysplasia type II" available on OpenNeuro.

## Background

In the realm of epilepsy imaging, the automated segmentation of Focal Cortical Dysplasias (FCDs) is crucial. FCDs, often undetected through conventional MRI analysis, require precise identification to significantly improve patient outcomes. Surgical removal of these dysplastic cortical areas has a high rate of success, frequently rendering patients seizure-free. The publication of our dataset aims to enhance computer-aided detection of FCDs, facilitating the refinement of existing algorithms and the innovation of new methodologies. This dataset encompasses T1 and FLAIR weighted MRI scans, meticulously annotated lesion masks, and pertinent clinical data, setting a new benchmark for research in the field.

## Dataset Information

The dataset citation is as follows:

```
@dataset{ds004199:1.0.5,
  author = {Fabiane Schuch and Lennart Walger and Matthias Schmitz and Bastian David and Tobias Bauer and Antonia Harms and Laura Fischbach and Freya Schulte and Martin Schidlowski and Johannes Reiter and Felix Bitzer and Randi von Wrede and Attila Rácz and Tobias Baumgartner and Valeri Borger and Matthias Schneider and Achim Flender and Albert Becker and Hartmut Vatter and Bernd Weber and Louisa Specht-Riemenschneider and Alexander Radbruch and Rainer Surges and Theodor Rüber},
  title = {"An open presurgery MRI dataset of people with epilepsy and focal cortical dysplasia type II"},
  year = {2023},
  doi = {doi:10.18112/openneuro.ds004199.v1.0.5},
  publisher = {OpenNeuro}
}
```

## Project Structure (so far)

```bash
Epilepsy-Dsiplasia-focal-Segmentation 
├─ bash-scripts
│  ├─ comprobar-dataset
│  │  └─ dataset-integrity.sh
│  ├─ limpiar-carpetas
│  │  └─ limpiar-carpetas.sh
│  ├─ organizar-estudios
│  │  └─ organizar_estudios.sh
│  ├─ pacientes-test
│  │  ├─ pacientes-test.txt
│  │  └─ pacientes_test.sh
│  └─ pacientes-validos
│     ├─ pacientes-validos.txt
│     └─ pacientes_validos.sh
├─ info-files
│  ├─ conda-list.txt
│  ├─ excluir-slices.txt
│  └─ t2flair-study
│     ├─ config.yaml
│     ├─ holdout
│     │  ├─ test_files.txt
│     │  ├─ train_files.txt
│     │  └─ val_files.txt
│     ├─ images
│     │  └─ dataset_distribution.png
│     ├─ roi-slices.txt
│     └─ specifications
│        ├─ images.txt
│        └─ labels.txt
├─ jupyter-notebooks
│  └─ explore_functions.ipynb
├─ src
│  ├─ __init__.py
│  ├─ data_augmentation.py
│  ├─ data_explore.py
│  ├─ data_loader.py
│  ├─ data_pipeline.py
│  ├─ yolov8_train.py
│  ├─ yolov8n-seg.pt
│  └─ yolov8n.pt
└─ torch_env.yml
```
