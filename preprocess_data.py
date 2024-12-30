#!/usr/bin/env python3
"""
preprocess_data.py

A script to preprocess pancreas MRI scans + masks by:
  1) Loading multi-frame DICOM using pydicom.
  2) Clipping to [p0.5, p99.5] of non-zero voxel intensities.
  3) Normalizing each volume to mean=0, std=1.
  4) Resampling in-plane to (256 x 256), preserving Z slices.
  5) Saving results as NIfTI (.nii.gz).

Requires:
  - pydicom
  - SimpleITK
  - numpy
Tested with Python 3.8+.
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk


def clip_percentile(volume: np.ndarray, low_percent=0.5, high_percent=99.5):
    """
    Clip the volume's non-zero intensities to [p0.5, p99.5].
    Returns clipped volume.
    """
    nonzero = volume[volume > 0]
    if nonzero.size < 10:
        # If nearly empty, skip percentile-based clipping
        return volume

    low_val = np.percentile(nonzero, low_percent)
    high_val = np.percentile(nonzero, high_percent)

    return np.clip(volume, low_val, high_val)


def preprocess_volume(mri_path: Path, mask_path: Path,
                      out_dir: Path, target_size=(256, 256),
                      low_percent=0.5, high_percent=99.5) -> bool:
    """
    Preprocess one case's (mri.dcm, mask.dcm).

    Steps:
      1) Load MRI and mask via pydicom.
      2) Clip intensities to [p0.5, p99.5].
      3) Mean-std normalize.
      4) Resample in-plane to target_size, keep Z same.
      5) Save NIfTI: volume.nii.gz and mask.nii.gz.

    Return:
      True if successful, False if shape mismatch or other error.
    """
    # 1) Load DICOM
    try:
        mri_dcm = pydicom.dcmread(str(mri_path))
        mask_dcm= pydicom.dcmread(str(mask_path))
    except Exception as e:
        print(f"[ERROR] Could not read DICOM for {mri_path.name}: {e}")
        return False

    mri_arr = mri_dcm.pixel_array.astype(np.float32)
    mask_arr= mask_dcm.pixel_array.astype(np.float32)

    # Remove extra dims if any, e.g. shape (1, 30, 256, 256)
    mri_arr = np.squeeze(mri_arr)
    mask_arr= np.squeeze(mask_arr)

    # Check shape mismatch
    if mri_arr.shape != mask_arr.shape:
        print(f"[WARNING] Shape mismatch: MRI={mri_arr.shape}, "
              f"Mask={mask_arr.shape}. Skipping {mri_path.parent.name}")
        return False

    # 2) Clip intensities (only MRI)
    mri_arr = clip_percentile(mri_arr, low_percent, high_percent)

    # 3) Mean-std normalization (only on non-zero)
    nonzero = mri_arr[mri_arr != 0]
    if nonzero.size > 0:
        mean_val = nonzero.mean()
        std_val  = nonzero.std()
    else:
        # fallback if empty
        mean_val = mri_arr.mean()
        std_val  = mri_arr.std()
    if std_val < 1e-6:
        std_val = 1e-6

    mri_arr = (mri_arr - mean_val) / std_val

    # Convert mask from {0,65535} => {0,1}
    # or any large values => 1
    mask_arr = (mask_arr > 0).astype(np.float32)

    # Convert to SITK images
    # SITK expects shape (Z, Y, X) => we interpret mri_arr as [Z,H,W].
    # If your array is (Z,H,W), SITK wants (Z,H,W) => done by "GetImageFromArray".
    img_sitk  = sitk.GetImageFromArray(mri_arr)
    mask_sitk = sitk.GetImageFromArray(mask_arr)

    # For in-plane resample => keep Z dimension the same
    original_size = img_sitk.GetSize()  # (width, height, depth) => SITK is (X, Y, Z)
    # We interpret that as X= W, Y= H, Z= # slices
    # So SITK size is (W, H, Z). We'll keep Z the same, resample W,H => target_size.

    new_W, new_H = target_size
    old_W, old_H, old_Z = original_size
    # We'll do the resampling so final => (new_W, new_H, old_Z).

    # We can guess the original spacing as (1,1,1) for now or read from DICOM tags
    spacing_x = 1.0
    spacing_y = 1.0
    spacing_z = 1.0
    if "PixelSpacing" in mri_dcm:
        # PixelSpacing often is [row_spacing, col_spacing], i.e. (dy, dx).
        # SITK ordering is (dx, dy). This is often reversed, so we read carefully:
        # Some scanners store it as (dr, dc).
        ps = mri_dcm.PixelSpacing
        if len(ps) == 2:
            spacing_y = float(ps[0])  # row spacing
            spacing_x = float(ps[1])  # column spacing
    if hasattr(mri_dcm, "SliceThickness"):
        spacing_z = float(mri_dcm.SliceThickness)

    img_sitk.SetSpacing((spacing_x, spacing_y, spacing_z))
    mask_sitk.SetSpacing((spacing_x, spacing_y, spacing_z))

    # Physical size = old_size * spacing
    orig_phys_size_x = old_W * spacing_x
    orig_phys_size_y = old_H * spacing_y
    # We'll keep Z dimension
    # new spacing in-plane = old_phys_size / new_size
    new_spacing_x = orig_phys_size_x / new_W
    new_spacing_y = orig_phys_size_y / new_H
    new_spacing_z = spacing_z  # unchanged

    # Prepare resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing((new_spacing_x, new_spacing_y, new_spacing_z))
    resampler.SetOutputOrigin(img_sitk.GetOrigin())
    resampler.SetOutputDirection(img_sitk.GetDirection())

    # We'll resample to new_size = (new_W, new_H, old_Z).
    # SITK wants (sizeX, sizeY, sizeZ)
    resampler.SetSize((new_W, new_H, old_Z))

    # Interpolate MRI with linear
    resampler.SetInterpolator(sitk.sitkLinear)
    img_256 = resampler.Execute(img_sitk)

    # Interpolate mask with nearest neighbor
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    mask_256 = resampler.Execute(mask_sitk)

    # Save as NIfTI
    out_dir.mkdir(parents=True, exist_ok=True)
    img_out  = out_dir / "volume.nii.gz"
    mask_out = out_dir / "mask.nii.gz"

    sitk.WriteImage(img_256, str(img_out))
    sitk.WriteImage(mask_256, str(mask_out))

    print(f"[INFO] Saved preprocessed => {img_out}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True,
                        help="Path to the input data directory containing subdirs with mri.dcm & mask.dcm.")
    parser.add_argument("--output_dir", required=True,
                        help="Where to save the preprocessed NIfTI output.")
    parser.add_argument("--low_percent", type=float, default=0.5,
                        help="Lower percentile for clipping non-zero intensities.")
    parser.add_argument("--high_percent", type=float, default=99.5,
                        help="Upper percentile for clipping non-zero intensities.")
    parser.add_argument("--target_size", type=int, default=256,
                        help="In-plane size to resample (H,W). Default=256 => 256x256.")
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"[ERROR] Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Traverse each case directory
    case_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    if not case_dirs:
        print(f"[WARNING] No subdirectories found under {input_dir}.")

    print(f"=== Starting Preprocessing ===")
    print(f" Input:  {input_dir}")
    print(f" Output: {output_dir}")
    print(f" Will resample in-plane to size=({args.target_size},{args.target_size}), keep Z the same.")
    print(f" Clip: p{args.low_percent} => p{args.high_percent}")

    processed_count = 0
    for case_dir in sorted(case_dirs):
        mri_path = case_dir / "mri.dcm"
        mask_path= case_dir / "mask.dcm"
        if not (mri_path.exists() and mask_path.exists()):
            print(f"[WARNING] Skipping {case_dir.name}, missing mri.dcm or mask.dcm.")
            continue

        out_subdir = output_dir / case_dir.name
        success = preprocess_volume(mri_path, mask_path, out_subdir,
                                    target_size=(args.target_size, args.target_size),
                                    low_percent=args.low_percent,
                                    high_percent=args.high_percent)
        if success:
            processed_count += 1

    print(f"\n[DONE] Preprocessed {processed_count} cases. "
          f"Results in => {output_dir}")
    sys.exit(0)


if __name__ == "__main__":
    main()
