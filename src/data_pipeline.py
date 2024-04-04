import data_loader
import data_augmentation
import data_explore

def main():
    dl_instance = data_loader.DataLoader(dataset_path="/home/mariopasc/Python/Datasets/epilepsy")
    dataaug_instance = data_augmentation.DataAugmentation(study_path="/home/mariopasc/Python/Datasets/ds-epilepsy/T2FLAIR",
                                        roi_path="/home/mariopasc/Python/Datasets/ds-epilepsy/ROI_T2")
    holdout_instance = data_loader.HoldOut(dataset_path="/home/mariopasc/Python/Datasets/ds-epilepsy",
                               study_name="T2FLAIR", roi_study="ROI_T2", val_percent=.25, test_percent=.1)
    dataaug_instance.apply_augmentations(augmentation_types=["brightness", "gamma"], num_patients=85)
    holdout_instance.organizeAugmentation()
    holdout_instance.holdout()
    results_path = "/home/mariopasc/Python/Projects/BSC_final/epilepsy-displasia-focal-segmentation/images"
    data_explore.analyze_dataset("/home/mariopasc/Python/Datasets/T2FLAIR-ds-epilepsy", results_path)

if __name__ == "__main__":
    main()
