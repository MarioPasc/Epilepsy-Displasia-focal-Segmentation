import data_loader
import data_augmentation
import data_explore
import time
import tqdm

def main():
    # Declaramos variables
    slices_excluir = ("/home/mariopasc/Python/Projects/BSC_final/epilepsy-displasia-focal-segmentation/info-files"
                      "/excluir-slices.txt")
    roi_slices_txt = ("/home/mariopasc/Python/Projects/BSC_final/epilepsy-displasia-focal-segmentation/info-files"
                      "/t2flair-study/roi-slices.txt")
    t2flair_path = "/home/mariopasc/Python/Datasets/ds-epilepsy/T2flair-study"
    t1w_path = "/home/mariopasc/Python/Datasets/ds-epilepsy/T1w-study"
    roi_path = "/home/mariopasc/Python/Datasets/ds-epilepsy/roi"

    t2flair_nii_path = "/home/mariopasc/Python/Datasets/ds-epilepsy/T2flair-study-nii"
    t1w_nii_path = "/home/mariopasc/Python/Datasets/ds-epilepsy/T1w-study-nii"
    roi_nii_path = "/home/mariopasc/Python/Datasets/ds-epilepsy/roi-nii"

    # Todas las variables que se declaran ahora deben ser añadidas en los ficheros .txt de especificación que se pasan
    # como parámetro a las funciones convert_nii_image_holdout y extract_roi_contours.
    """ 
    t2flair_im_train_path = "/home/mariopasc/Python/Datasets/t2flair-yolov8-ds/images/train"
    t2flair_im_val_path = "/home/mariopasc/Python/Datasets/t2flair-yolov8-ds/images/val"
    t2flair_label_train_path = "/home/mariopasc/Python/Datasets/t2flair-yolov8-ds/labels/train"
    t2flair_label_train_val = "/home/mariopasc/Python/Datasets/t2flair-yolov8-ds/labels/val"

    t1w_im_train_path = "/home/mariopasc/Python/Datasets/t1w-yolov8-ds/images/train"
    t1w_im_val_path = "/home/mariopasc/Python/Datasets/t1w-yolov8-ds/images/val"
    t1w_label_train_path = "/home/mariopasc/Python/Datasets/t1w-yolov8-ds/labels/train"
    t1w_label_train_val = "/home/mariopasc/Python/Datasets/t1w-yolov8-ds/labels/val"
    """

    # =========== CONVERSIÓN ESTUDIO A NII ===========
    # IMAGES
    # Primero tener que convertir nuestro estudio a ficheros .nii atómicos para cada imagen
    print("🔸Convirtiendo el estudio T2 a formato .nii")
    data_loader.study_to_nii(study_path=t2flair_path,
                             save_path=t2flair_nii_path,
                             path_excluir=slices_excluir)
    print("✅ Imágenes convertidas a .nii")
    print("🔸Convirtiendo el estudio ROI a formato .nii")
    data_loader.study_to_nii(study_path=roi_path,
                             save_path=roi_nii_path,
                             path_excluir=slices_excluir)
    print("✅ ROI convertido a .nii")
    time.sleep(5)
    # ==================================================

    # =============== DATA AUGMENTATION ================

    # Nos interesa hacer un oversampling de la clase "lesion". Por ello, aumentaremos el número de imágenes con un
    # ROI asociado haciendo ligeras modificaciones.
    roi_txt = ("/home/mariopasc/Python/Projects/BSC_final/epilepsy-displasia-focal-segmentation/info-files"
               "/t2flair-study/roi-slices.txt")
    print("🔸Comenzamos aumento de datos para cortes con ROI")
    data_augmentation.save_slices_with_roi(input_dir="/home/mariopasc/Python/Datasets/ds-epilepsy/roi",
                                           output_txt=roi_txt)
    time.sleep(5)
    # Aplicamos ahora unos aumento de datos a solo los cortes con ROI
    data_augmentation.apply_augmentation(num_images=1500,
                                         input_folder=t2flair_nii_path,
                                         roi_txt_file=roi_slices_txt,
                                         exclusion_file=slices_excluir,
                                         roi_folder=roi_nii_path,
                                         augmentation_method="TRANSLATION")
    data_augmentation.apply_augmentation(num_images=1500,
                                         input_folder=t2flair_nii_path,
                                         roi_txt_file=roi_slices_txt,
                                         exclusion_file=slices_excluir,
                                         roi_folder=roi_nii_path,
                                         augmentation_method="ROTATION")
    print("✅ Aumento de datos completado")
    # =========================================

    # =============== HOLDOUT =================
    # Ahora, dado esa carpeta con los ficheros, le realizamos un holdout y guardamos los archivos
    # .txt train val y test en una carpeta
    print("🔸Comenzamos Holdout")
    data_loader.holdout_nii_images(folder_path=t2flair_nii_path,
                                   val_percent=0.3,
                                   test_percent=0.1,
                                   output_path="/home/mariopasc/Python/Projects/BSC_final/epilepsy-displasia-focal"
                                               "-segmentation/info-files/t2flair-study/holdout")
    print("✅ Holdout realizado")
    # =========================================

    # =============== NII TO PNG ==============
    # Finalmente, damos:
    # 1. La carpeta con las imágenes .nii
    # 2. La ruta a los ficheros .txt que están destinados a train, a val o a test
    # 3. El formato que compresión que queremos
    print("🔸Comenzamos conversión a JPG")
    data_loader.convert_nii_image_holdout(
        input_txt="/home/mariopasc/Python/Projects/BSC_final/epilepsy-displasia-focal-segmentation/info-files/t2flair"
                  "-study/specifications/images.txt")
    print("✅ Imágenes convertidas a PNG y recolocadas")
    # =========================================

    # ============= LABELS TO YOLO ============
    # Primero tener que convertir nuestro estudio a ficheros .nii atómicos para cada imagen
    print("🔸Comenzamos conversión ROI a YOLO txt")
    time.sleep(5)
    data_loader.extract_roi_contours(input_txt="/home/mariopasc/Python/Projects/BSC_final/epilepsy-displasia-focal"
                                               "-segmentation/info-files/t2flair-study/specifications/labels.txt")
    print("✅ ROI convertido a formato YOLOv8")

    # NOTA: RECUERDA!! No todas las imágenes tienen un label.txt asociado, si una imagen no tiene un contorno asociado
    # al ROI, entonces no se crea un .txt correspondiente. 1967
    # =========================================

    # ============ VISUALIZACION ==============

    data_explore.analyze_dataset(folder_paths=[
        "/home/mariopasc/Python/Datasets/t2flair-yolov8-ds/images/train",
        "/home/mariopasc/Python/Datasets/t2flair-yolov8-ds/images/val",
        "/home/mariopasc/Python/Datasets/t2flair-yolov8-ds/images/test"
    ], label_file_path="/home/mariopasc/Python/Projects/BSC_final/epilepsy-displasia-focal-segmentation/info-files"
                       "/t2flair-study/roi-slices.txt",
    save_path="/home/mariopasc/Python/Projects/BSC_final/epilepsy-displasia-focal-segmentation/info-files/t2flair"
              "-study/images")

    # =========================================


if __name__ == "__main__":
    main()
