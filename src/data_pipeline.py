import data_loader
import data_augmentation
import data_explore

def main():
    dl_instance = data_loader.DataLoader(dataset_path="/home/mariopasc/Python/Datasets/epilepsy")
    holdout_instance = data_loader.HoldOut(dataset_path="/home/mariopasc/Python/Datasets/ds-epilepsy",
                               study_name="T2FLAIR", roi_study="ROI_T2", val_percent=.2, test_percent=.2)
    print(f"Test: {holdout_instance.test_set}")
    dataaug_instance = data_augmentation.DataAugmentation(study_path="/home/mariopasc/Python/Datasets/ds-epilepsy/T2FLAIR",
                                        roi_path="/home/mariopasc/Python/Datasets/ds-epilepsy/ROI_T2", HoldOutInstance=holdout_instance)
    
    dataaug_instance.apply_augmentations(augmentation_types=["brightness", "gamma"], num_patients=75)
    holdout_instance.organizeAugmentation()

    print()

    results_path = "/home/mariopasc/Python/Projects/BSC_final/epilepsy-displasia-focal-segmentation/images"
    data_explore.analyze_dataset("/home/mariopasc/Python/Datasets/T2FLAIR-ds-epilepsy", results_path)

if __name__ == "__main__":
    main()
