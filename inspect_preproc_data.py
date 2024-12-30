#!/usr/bin/env python3

"""
inspect_preproc_data.py

Inspect a directory of preprocessed pancreas data.
Each subdirectory is expected to have:
  volume.nii.gz
  mask.nii.gz

We compute:
  - shape info for vol & mask
  - min, max, mean, std of volume intensities
  - mask sum, fraction of non-zero, bounding box of mask
  - #nonempty slices, #completely empty slices, etc.

Outputs a summary CSV or prints to screen.
"""

import os
import sys
import argparse
import json
from pathlib import Path

import numpy as np
import nibabel as nib

def compute_bounding_box(mask_3d: np.ndarray) -> tuple:
    """
    Compute bounding box (z_min,z_max, y_min,y_max, x_min,x_max)
    for a 3D binary mask.  If mask is all zeros => return None or a
    sentinel.
    """
    nonzero = np.argwhere(mask_3d > 0.5)
    if nonzero.size == 0:
        return None
    zmin, ymin, xmin = nonzero.min(axis=0)
    zmax, ymax, xmax = nonzero.max(axis=0)
    return (int(zmin), int(zmax), int(ymin), int(ymax), int(xmin), int(xmax))

def main():
    global np
    parser = argparse.ArgumentParser(description="Inspect preprocessed pancreas data")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the preprocessed data directory. "
                             "It should contain subdirs, each with volume.nii.gz and mask.nii.gz.")
    parser.add_argument("--output_path", type=str, default="inspection_results.json",
                        help="Where to store the output JSON or CSV summary.")
    parser.add_argument("--summarize", action="store_true",
                        help="If set, we only print the final summary rather than per-case info.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"[ERROR] {data_dir} is not a directory.")
        sys.exit(1)

    # Gather all subdirectories
    case_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])

    results_list = []

    for case_dir in case_dirs:
        vol_path = case_dir / "volume.nii.gz"
        mask_path= case_dir / "mask.nii.gz"
        if not (vol_path.exists() and mask_path.exists()):
            print(f"[WARNING] Skipping {case_dir.name} - missing volume.nii.gz or mask.nii.gz")
            continue

        # Load with nibabel
        vol_nib  = nib.load(str(vol_path))
        mask_nib = nib.load(str(mask_path))

        vol_data  = vol_nib.get_fdata(dtype=np.float32)  # shape [Z,Y,X] or [Z,H,W]
        mask_data = mask_nib.get_fdata(dtype=np.float32) # shape same as vol_data
        # Binarize (just in case)
        mask_data = (mask_data > 0.5).astype(np.float32)

        shape_vol  = vol_data.shape
        shape_mask = mask_data.shape

        # basic intensity stats
        vol_min = float(vol_data.min())
        vol_max = float(vol_data.max())
        vol_mean= float(vol_data.mean())
        vol_std = float(vol_data.std())

        # mask stats
        mask_sum = float(mask_data.sum())  # total #voxels
        fraction_nonzero = mask_sum / float(mask_data.size)
        
        # bounding box
        bbox_6d = compute_bounding_box(mask_data)  # (zmin,zmax,ymin,ymax,xmin,xmax) or None

        # #nonempty slices
        # We'll count how many slices (along Z) have at least some mask
        z_dim = shape_mask[0] if len(shape_mask)>2 else 1
        nonempty_slices = 0
        for z_idx in range(z_dim):
            slice_mask = mask_data[z_idx,...]
            if slice_mask.sum()>0:
                nonempty_slices += 1

        # store in results
        case_info = {
            "case_name":      case_dir.name,
            "vol_shape":      shape_vol,
            "mask_shape":     shape_mask,
            "vol_min":        vol_min,
            "vol_max":        vol_max,
            "vol_mean":       vol_mean,
            "vol_std":        vol_std,
            "mask_sum":       mask_sum,
            "mask_fraction":  fraction_nonzero,
            "nonempty_slices": nonempty_slices,
            "bbox_z_y_x":     bbox_6d,  # None if empty
        }
        results_list.append(case_info)

    # Write out JSON summary
    # Or you could choose CSV. We'll do JSON for convenience.
    with open(args.output_path, "w") as f:
        json.dump(results_list, f, indent=2)

    if not args.summarize:
        # If user wants all info, we print a summary line per case
        for r in results_list:
            print(f"Case={r['case_name']} shape={r['vol_shape']} mask_shape={r['mask_shape']} "
                  f"vol_min={r['vol_min']:.3f} vol_max={r['vol_max']:.3f} mean={r['vol_mean']:.3f} std={r['vol_std']:.3f} "
                  f"mask_sum={r['mask_sum']:.1f} frac={r['mask_fraction']:.5f} nonempty_slices={r['nonempty_slices']} "
                  f"bbox={r['bbox_z_y_x']}")
    else:
        # If summarize => e.g. we might compute overall means
        # We'll do something simple: #cases, average mask_fraction, etc.
        import numpy as np
        mask_fracs = [r['mask_fraction'] for r in results_list]
        avg_frac = np.mean(mask_fracs) if len(mask_fracs)>0 else 0
        print(f"[SUMMARY] Found {len(results_list)} valid cases. "
              f"Average mask fraction={avg_frac:.6f}. "
              f"See {args.output_path} for details.")


if __name__=="__main__":
    main()
