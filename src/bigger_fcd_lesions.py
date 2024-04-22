import numpy as np
import cv2
import os
from tqdm import tqdm
import nibabel as nib
from typing import List
import glob

class BiggerLesions:

    def __init__(self, study_path: str, roi_path: str, kernel_size:int, iterations:int) -> None:
        self.study_path = study_path
        self.roi_path = roi_path

        self.kernel_size = kernel_size
        self.iterations = iterations

        self.output_image_folder = f"BiggerFCD_{self.study_path.split('/')[-1]}"
        self.output_roi_folder = f"BiggerFCD_ROI{self.study_path.split('/')[-1]}"
        os.makedirs(os.path.join(self.study_path, "..", self.output_image_folder), exist_ok=True)
        os.makedirs(os.path.join(self.study_path, "..", self.output_roi_folder), exist_ok=True)

    def _applyDilation(self, image_slice:np.ndarray, mask_slice:np.ndarray) -> List[np.ndarray]:
        # Extract the corresponding lesion from the original image
        lesion_region = cv2.bitwise_and(image_slice, image_slice, mask=mask_slice)

        # Dilate the mask
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        dilated_lesion = cv2.dilate(lesion_region, kernel, iterations=self.iterations)

        # Create the result image by copying the original slice
        result_image = image_slice.copy()

        # Paste the dilated lesion onto the result image
        result_image[dilated_lesion > 0] = dilated_lesion[dilated_lesion > 0]

        return [result_image, dilated_lesion]
    
    def _getCorrespondingROI(self, image_name:str) -> str:
        split_by_sub = image_name.strip(".nii.gz").split("_")
        id_patient = split_by_sub[0]
        matching_files = glob.glob(os.path.join(self.roi_path, f"{id_patient}*.nii.gz"))
        if len(matching_files) > 0:
            return matching_files[0]
        else:
            raise FileNotFoundError(f"No matching ROI file found for {image_name}")

    def generateBiggerLesionsDataset(self) -> None:
        for niigzFile in tqdm(os.listdir(self.study_path), colour="red", desc="Generated Bigger FCD Lesions"):
            if niigzFile.endswith(".nii.gz"):
                image_path = os.path.join(self.study_path, niigzFile)
                # Get the corresponding ROI file
                roi_file_path = self._getCorrespondingROI(niigzFile)
                
                # Load the image and mask
                image = np.uint8(nib.load(image_path).get_fdata())
                mask = np.uint8(nib.load(roi_file_path).get_fdata())
                
                # Initialize arrays to store the processed volumes
                volume_images = []
                volume_masks = []
                
                # Process each slice
                for slice_index in range(image.shape[2]):
                    image_slice = image[:, :, slice_index]
                    mask_slice = mask[:, :, slice_index]
                    result = self._applyDilation(image_slice, mask_slice)
                    image_biggerLesion = result[0]
                    roi_biggerLesion = result[1]
                    volume_images.append(image_biggerLesion)
                    volume_masks.append(roi_biggerLesion)
                
                # Convert the lists to numpy arrays
                volume_images = np.array(volume_images)
                volume_masks = np.array(volume_masks)
                
                # Transpose the arrays to have the correct dimensions (x, y, z)
                volume_images = volume_images.transpose((1, 2, 0))
                volume_masks = volume_masks.transpose((1, 2, 0))
                
                # Save the processed image and mask
                output_image_path = os.path.join(self.study_path, "..", self.output_image_folder, niigzFile)
                output_roi_path = os.path.join(self.study_path, "..", self.output_roi_folder, os.path.basename(roi_file_path))
                nib.save(nib.Nifti1Image(volume_images, affine=None), output_image_path)
                nib.save(nib.Nifti1Image(volume_masks, affine=None), output_roi_path)
