import data_loader
import data_augmentation

def main():
    dl_instance = data_loader.DataLoader(dataset_path="/home/mariopasc/Python/Datasets/epilepsy")
    dataaug_instance = data_augmentation.DataAugmentation(study_path="/home/mariopasc/Python/Datasets/ds-epilepsy/T2FLAIR",
                                        roi_path="/home/mariopasc/Python/Datasets/ds-epilepsy/ROI_T2")
    dataaug_instance.apply_augmentations(augmentation_types=["brightness", "gamma"],
                                         num_patients=100)
    holdout_instance = data_loader.HoldOut(dataset_path="/home/mariopasc/Python/Datasets/ds-epilepsy",
                               study_name="T2FLAIR", roi_study="ROI_T2", val_percent=.2, test_percent=.1)

if __name__ == "__main__":
    main()
