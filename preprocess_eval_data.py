#!/usr/bin/env python3
"""
preprocess_eval_data.py

**Purpose**:
    1. Reads multi-frame DICOM files (mri.dcm, mask.dcm) from
       /scratch/09999/aymanmahfuz/pancreas_eval_data/
       (each subdirectory has mri.dcm, mask.dcm)
    2. Checks for shape mismatches or read errors. Those cases are skipped.
    3. Converts valid pairs to NIfTI and saves them in a structure suitable
       for nnU-Net *inference*:
         /scratch/09999/aymanmahfuz/pancreas_eval_data_preprocessed
           ├── imagesTs/
           └── labelsTs/
       Each MRI --> {case_id}_0000.nii.gz (so we can run nnUNet_predict)
       Each mask --> {case_id}.nii.gz (for later Dice comparison)
    4. Normalizes mask values if needed (e.g., from 65535 to 1).
    5. Output is strictly for evaluation data (imagesTs, labelsTs), not training.

**Usage**:
    python preprocess_eval_data.py
"""

import os
import SimpleITK as sitk
import numpy as np

# Source directory with subfolders (each has mri.dcm, mask.dcm)
SOURCE_EVAL_DIR = "/scratch/09999/aymanmahfuz/pancreas_eval_data"

# Output directory for nnU-Net inference
TARGET_EVAL_DIR = "/scratch/09999/aymanmahfuz/pancreas_eval_data_preprocessed"

IMAGES_TS_DIR = os.path.join(TARGET_EVAL_DIR, "imagesTs")
LABELS_TS_DIR = os.path.join(TARGET_EVAL_DIR, "labelsTs")

def read_dicom_volume(dicom_path):
    """
    Reads a multi-frame DICOM file using SimpleITK and returns the image object.
    Raises an exception on error (e.g., shape mismatch or read fails).
    """
    reader = sitk.ImageFileReader()
    reader.SetFileName(dicom_path)
    image = reader.Execute()
    return image

def convert_and_save_eval(mri_path, mask_path, case_id):
    """
    Converts the DICOM volume to NIfTI, checks shape mismatch, and saves.
    Also normalizes mask values from 65535 to 1 if needed.
    For inference, we place MRI in imagesTs (case_id_0000.nii.gz)
    and mask in labelsTs (case_id.nii.gz) for later metric computation.
    """
    # Read MRI
    mri_image = read_dicom_volume(mri_path)
    mri_array = sitk.GetArrayFromImage(mri_image)
    mri_shape = mri_array.shape

    # Read mask
    mask_image = read_dicom_volume(mask_path)
    mask_array = sitk.GetArrayFromImage(mask_image)
    mask_shape = mask_array.shape

    if mri_shape != mask_shape:
        raise ValueError(f"Shape mismatch: MRI={mri_shape} vs MASK={mask_shape}")

    # Convert mask >0 to 1
    mask_array[mask_array > 0] = 1
    mask_image = sitk.GetImageFromArray(mask_array)
    mask_image.CopyInformation(mri_image)

    # imagesTs: {case_id}_0000.nii.gz
    out_mri_path = os.path.join(IMAGES_TS_DIR, f"{case_id}_0000.nii.gz")
    sitk.WriteImage(mri_image, out_mri_path)

    # labelsTs: {case_id}.nii.gz
    out_mask_path = os.path.join(LABELS_TS_DIR, f"{case_id}.nii.gz")
    sitk.WriteImage(mask_image, out_mask_path)

def main():
    # Make output dirs
    os.makedirs(IMAGES_TS_DIR, exist_ok=True)
    os.makedirs(LABELS_TS_DIR, exist_ok=True)

    subdirs = [d for d in os.listdir(SOURCE_EVAL_DIR)
               if os.path.isdir(os.path.join(SOURCE_EVAL_DIR, d))]

    converted = 0
    for subdir in subdirs:
        mri_path = os.path.join(SOURCE_EVAL_DIR, subdir, "mri.dcm")
        mask_path = os.path.join(SOURCE_EVAL_DIR, subdir, "mask.dcm")

        if not os.path.exists(mri_path) or not os.path.exists(mask_path):
            print(f"[SKIP] {subdir}: missing mri.dcm or mask.dcm.")
            continue

        # Use the subdirectory name as case_id
        case_id = subdir

        try:
            convert_and_save_eval(mri_path, mask_path, case_id)
            print(f"[OK] Converted {subdir} --> {case_id}")
            converted += 1
        except Exception as e:
            print(f"[SKIP] {subdir} due to error: {e}")
            continue

    print(f"\nAll done. Successfully converted {converted} evaluation cases.")

if __name__ == "__main__":
    main()
