#!/usr/bin/env python3

"""
Check domain differences between training data and evaluation data.
- Iterates over each subfolder containing {mri.dcm, mask.dcm}.
- Prints out or saves shape, voxel stats, coverage, basic DICOM metadata, etc.
- Summarizes results at the end (unique shapes, min/max intensities, etc.).
"""

import os
import sys
import pydicom
import numpy as np
from pathlib import Path
import argparse
import json
from collections import defaultdict

def analyze_folder(case_dir):
    """
    Analyze a single folder containing 'mri.dcm' and 'mask.dcm'.
    Returns a dictionary with shape, min/max, coverage, metadata, etc.
    """
    case_dir = Path(case_dir)
    mri_path = case_dir / 'mri.dcm'
    mask_path = case_dir / 'mask.dcm'
    info = {
        "case_name": case_dir.name,
        "mri_exists": mri_path.exists(),
        "mask_exists": mask_path.exists(),
        "mri_shape": None,
        "mask_shape": None,
        "mri_min": None,
        "mri_max": None,
        "mask_min": None,
        "mask_max": None,
        "mask_unique_vals": None,
        "mask_coverage": None,
        "dicom_spacing": None,        # PixelSpacing
        "dicom_thickness": None,      # SliceThickness
        "dicom_series_desc": None,    # SeriesDescription (often has scan info)
        "dicom_manufacturer": None,   # Manufacturer
        # Add more relevant tags if needed
        "error": None
    }

    if not (mri_path.exists() and mask_path.exists()):
        info["error"] = "Missing mri.dcm or mask.dcm"
        return info

    try:
        # Read the MRI DICOM
        mri_dcm = pydicom.dcmread(str(mri_path))
        mri_arr = mri_dcm.pixel_array  # shape => (T,H,W) or sometimes (H,W) if single slice
        mri_arr = np.squeeze(mri_arr)  # remove extra dims if any
        # Check if it’s 2D vs. 3D:
        if mri_arr.ndim == 2:
            # e.g. shape => (H,W)
            T_vol = 1
            H, W = mri_arr.shape
            shape_str = f"(1, {H}, {W})"
        elif mri_arr.ndim == 3:
            T_vol, H, W = mri_arr.shape
            shape_str = f"({T_vol}, {H}, {W})"
        else:
            shape_str = str(mri_arr.shape)  # fallback

        info["mri_shape"] = shape_str
        info["mri_min"] = float(mri_arr.min())
        info["mri_max"] = float(mri_arr.max())

        # Basic DICOM tags
        if hasattr(mri_dcm, "PixelSpacing"):
            info["dicom_spacing"] = tuple(mri_dcm.PixelSpacing)
        if hasattr(mri_dcm, "SliceThickness"):
            info["dicom_thickness"] = str(mri_dcm.SliceThickness)
        if hasattr(mri_dcm, "SeriesDescription"):
            info["dicom_series_desc"] = str(mri_dcm.SeriesDescription)
        if hasattr(mri_dcm, "Manufacturer"):
            info["dicom_manufacturer"] = str(mri_dcm.Manufacturer)

        # Read the Mask DICOM
        mask_dcm = pydicom.dcmread(str(mask_path))
        mask_arr = mask_dcm.pixel_array.astype(np.float32)
        mask_arr = np.squeeze(mask_arr)

        # Compare shape
        if mask_arr.shape != mri_arr.shape:
            # shape mismatch
            info["error"] = f"Shape mismatch. MRI shape={mri_arr.shape}, Mask shape={mask_arr.shape}"

        # Summaries
        info["mask_shape"] = str(mask_arr.shape)
        info["mask_min"] = float(mask_arr.min())
        info["mask_max"] = float(mask_arr.max())

        # Unique values
        unique_vals = np.unique(mask_arr)
        # If you only want up to some max count:
        if len(unique_vals) > 20:
            unique_vals_str = f"{unique_vals[:20]}... (total {len(unique_vals)})"
        else:
            unique_vals_str = str(unique_vals)
        info["mask_unique_vals"] = unique_vals_str

        # Coverage = fraction of nonzero
        coverage = (mask_arr > 0).sum() / mask_arr.size
        info["mask_coverage"] = float(coverage)

    except Exception as e:
        info["error"] = f"Error reading or processing case: {e}"

    return info


def main(train_dir, eval_dir, output_json):
    """
    For each directory in train_dir and eval_dir, analyze subfolders.
    Collect stats, print summary, and optionally save to JSON.
    """
    train_dir = Path(train_dir)
    eval_dir = Path(eval_dir)

    # We'll store results in a list of dicts
    results_train = []
    results_eval = []

    # Analyze training data
    train_cases = [d for d in train_dir.iterdir() if d.is_dir()]
    print(f"\nAnalyzing training data: {len(train_cases)} cases under {train_dir}")
    for case_dir in train_cases:
        info = analyze_folder(case_dir)
        info["split"] = "train"
        results_train.append(info)

    # Analyze evaluation data
    eval_cases = [d for d in eval_dir.iterdir() if d.is_dir()]
    print(f"\nAnalyzing evaluation data: {len(eval_cases)} cases under {eval_dir}")
    for case_dir in eval_cases:
        info = analyze_folder(case_dir)
        info["split"] = "eval"
        results_eval.append(info)

    # Merge
    all_results = results_train + results_eval

    # Print short summary
    def short_case_summary(r):
        return (
            f"{r['split'].upper()} | {r['case_name']} | "
            f"MRI Shape={r['mri_shape']} | MRI Range=[{r['mri_min']:.1f},{r['mri_max']:.1f}] | "
            f"Mask Shape={r['mask_shape']} | Mask Coverage={r['mask_coverage']:.3f} | "
            f"Error={r['error']}"
        )

    print("\n=== Detailed Results ===")
    for r in all_results:
        print(short_case_summary(r))

    # Summaries: gather shapes, min, max, coverage
    # Example: unique shapes
    shapes_train = defaultdict(int)
    shapes_eval = defaultdict(int)
    for r in results_train:
        shapes_train[r["mri_shape"]] += 1
    for r in results_eval:
        shapes_eval[r["mri_shape"]] += 1

    print("\n=== Unique MRI shapes in TRAIN ===")
    for s, count in shapes_train.items():
        print(f"  Shape={s}  Count={count}")

    print("\n=== Unique MRI shapes in EVAL ===")
    for s, count in shapes_eval.items():
        print(f"  Shape={s}  Count={count}")

    # If you want to see how many had shape mismatch or other errors
    mismatches = [x for x in all_results if x["error"] is not None]
    print(f"\n# of Cases with errors or shape mismatch = {len(mismatches)}")
    for m in mismatches[:10]:
        print(f"  -> {m['split']}: {m['case_name']} error={m['error']}")

    # Save to JSON if requested
    if output_json:
        out_dict = {
            "all_results": all_results,
            "train_count": len(results_train),
            "eval_count": len(results_eval),
        }
        with open(output_json, "w") as f:
            json.dump(out_dict, f, indent=2)
        print(f"\nSaved full details to {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, required=True,
                        help="Path to the training data parent directory")
    parser.add_argument("--eval_dir", type=str, required=True,
                        help="Path to the evaluation data parent directory")
    parser.add_argument("--output_json", type=str, default="domain_analysis.json",
                        help="Optional JSON file to save all results")
    args = parser.parse_args()

    main(args.train_dir, args.eval_dir, args.output_json)
