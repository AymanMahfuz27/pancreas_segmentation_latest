#!/usr/bin/env python3
"""
prepare_nnunet_data.py

**Purpose**:
    1. Reads multi-frame DICOM files (mri.dcm, mask.dcm) from
       /scratch/09999/aymanmahfuz/pancreas_test_data
    2. Checks for shape mismatches or any read errors. Those cases are skipped.
    3. Converts valid pairs to NIfTI and saves them in a structure suitable for nnU-Net:
         /scratch/09999/aymanmahfuz/pancreas_test_data_preprocessed_paNSegNet/
           ├── imagesTr/
           └── labelsTr/
       Each MRI →  case_XXX_0000.nii.gz
       Each mask → case_XXX.nii.gz
    4. Normalizes mask values if needed (e.g., from 65535 to 1).

**Notes**:
    - If a shape mismatch or error occurs, we skip that case.
    - The final data will be used later for nnU-Net plan/preprocess.
    - Adjust indexing, naming, or any additional filtering logic as needed.

**Usage**:
    python prepare_nnunet_data.py

"""

import os
import SimpleITK as sitk
import numpy as np

# Original data directory with subfolders, each containing mri.dcm and mask.dcm
SOURCE_DIR = "/scratch/pancreas_test_data"

# New directory to store nnU-Net compatible data
TARGET_DIR = "/scratch/pancreas_test_data_preprocessed_paNSegNet"

# Subfolders for images (MRI) and labels (mask)
IMAGES_TR_DIR = os.path.join(TARGET_DIR, "imagesTr")
LABELS_TR_DIR = os.path.join(TARGET_DIR, "labelsTr")

def read_dicom_volume(dicom_path):
    """
    Reads a multi-frame DICOM file using SimpleITK and returns the image object.
    Raises exception on error (e.g., if shape mismatch or read fails).
    """
    reader = sitk.ImageFileReader()
    reader.SetFileName(dicom_path)
    image = reader.Execute()
    return image

def convert_and_save(mri_path, mask_path, case_id):
    """
    Converts the DICOM volume to NIfTI, checks shape mismatch, and saves.
    Also normalizes mask values from 65535 to 1 (if needed).
    """
    # Read MRI
    mri_image = read_dicom_volume(mri_path)
    mri_array = sitk.GetArrayFromImage(mri_image)  # shape: [Z, Y, X]
    mri_shape = mri_array.shape  # e.g., (30, 256, 256)

    # Read Mask
    mask_image = read_dicom_volume(mask_path)
    mask_array = sitk.GetArrayFromImage(mask_image)
    mask_shape = mask_array.shape

    # Check shape mismatch
    if mri_shape != mask_shape:
        raise ValueError(f"Shape mismatch: MRI={mri_shape} vs MASK={mask_shape}")

    # If mask contains 65535 for pancreas, convert to 1
    # We can do a threshold-based approach:
    # any voxel > 0 is set to 1.
    mask_array[mask_array > 0] = 1
    # Recreate the sitk Image for output
    mask_image = sitk.GetImageFromArray(mask_array)
    mask_image.CopyInformation(mri_image)  # keep spacing/origin consistent

    # Save MRI → imagesTr/case_XXX_0000.nii.gz
    out_mri_path = os.path.join(IMAGES_TR_DIR, f"{case_id}_0000.nii.gz")
    sitk.WriteImage(mri_image, out_mri_path)

    # Save Mask → labelsTr/case_XXX.nii.gz
    out_mask_path = os.path.join(LABELS_TR_DIR, f"{case_id}.nii.gz")
    sitk.WriteImage(mask_image, out_mask_path)


def main():
    # Create output dirs if they don't exist
    os.makedirs(IMAGES_TR_DIR, exist_ok=True)
    os.makedirs(LABELS_TR_DIR, exist_ok=True)

    subdirs = [d for d in os.listdir(SOURCE_DIR)
               if os.path.isdir(os.path.join(SOURCE_DIR, d))]

    case_count = 0
    for subdir in subdirs:
        mri_path = os.path.join(SOURCE_DIR, subdir, "mri.dcm")
        mask_path = os.path.join(SOURCE_DIR, subdir, "mask.dcm")

        # Quick check if both exist
        if not os.path.exists(mri_path) or not os.path.exists(mask_path):
            print(f"[SKIP] {subdir}: missing mri.dcm or mask.dcm.")
            continue

        case_id_str = f"case_{case_count:03d}"

        try:
            convert_and_save(mri_path, mask_path, case_id_str)
            print(f"[OK] Converted {subdir} → {case_id_str}")
            case_count += 1
        except Exception as e:
            print(f"[SKIP] {subdir}: {e}")
            continue

    print(f"\nAll done. Successfully converted {case_count} cases.")

if __name__ == "__main__":
    main()
