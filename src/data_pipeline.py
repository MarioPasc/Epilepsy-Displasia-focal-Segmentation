import data_loader
import data_augmentation
import data_explore
import generate_dsyolov8
import bigger_fcd_lesions
import os
def generateAugmentedDataset():
    dl_instance = data_loader.DataLoader(dataset_path="/home/mariopasc/Python/Datasets/epilepsy")
    holdout_instance = data_loader.HoldOut(dataset_path="/home/mariopasc/Python/Datasets/ds-epilepsy",
                               study_name="T2FLAIR", roi_study="ROI_T2", val_percent=.2, test_percent=.2)
    dataaug_instance = data_augmentation.DataAugmentation(study_path="/home/mariopasc/Python/Datasets/ds-epilepsy/T2FLAIR",
                                        roi_path="/home/mariopasc/Python/Datasets/ds-epilepsy/ROI_T2", HoldOutInstance=holdout_instance)
    
    dataaug_instance.apply_augmentations(augmentation_types=["brightness", "gamma"], num_patients=0)
    
    #print(f"Train set: {holdout_instance.train_set} \nVal set: {holdout_instance.val_set} \nTest set: {holdout_instance.test_set}")
    #print("======================")
    yolov8_ds_generator_instance = generate_dsyolov8.YOLOv8DatasetGenerator(HoldOutInstance=holdout_instance)
    #print(f"Train set: {yolov8_ds_generator_instance.train_set} \nVal set: {yolov8_ds_generator_instance.val_set} \nTest set: {yolov8_ds_generator_instance.test_set}")
    yolov8_ds_generator_instance.generate_yolo_ds()
    results_path = "/home/mariopasc/Python/Projects/BSC_final/epilepsy-displasia-focal-segmentation/images"
    data_explore.analyze_dataset("/home/mariopasc/Python/Datasets/T2FLAIR_yolov8_dataset", results_path)

def generateBiggerLesionsDataset():
    dataset_father_folder = "/home/mariopasc/Python/Datasets/epilepsy"
    dl_instance = data_loader.DataLoader(dataset_path=dataset_father_folder)
    biggerLesions = bigger_fcd_lesions.BiggerLesions(study_path=os.path.join("/home/mariopasc/Python/Datasets/ds-epilepsy", "T2FLAIR"),
                                                     roi_path=os.path.join("/home/mariopasc/Python/Datasets/ds-epilepsy", "ROI_T2"),
                                                     kernel_size=3,
                                                     iterations=2)
    biggerLesions.generateBiggerLesionsDataset()
    holdout_instance = data_loader.HoldOut(dataset_path="/home/mariopasc/Python/Datasets/ds-epilepsy",
                               study_name="BiggerFCD_T2FLAIR", roi_study="BiggerFCD_ROIT2FLAIR", val_percent=.2, test_percent=.2)
    yolov8_ds_generator_instance = generate_dsyolov8.YOLOv8DatasetGenerator(HoldOutInstance=holdout_instance)
    yolov8_ds_generator_instance.generate_yolo_ds()
    results_path = "/home/mariopasc/Python/Projects/BSC_final/epilepsy-displasia-focal-segmentation/images"
    data_explore.analyze_dataset("/home/mariopasc/Python/Datasets/BiggerFCD_T2FLAIR_yolov8_dataset", results_path)

def main():
    generateBiggerLesionsDataset()

if __name__ == "__main__":
    main()
