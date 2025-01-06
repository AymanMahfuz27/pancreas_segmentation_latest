#!/usr/bin/env python3
"""
evaluate_inference.py

**Purpose**:
    1. Runs nnUNet_predict on the preprocessed evaluation images (imagesTs/).
    2. Compares the predicted masks with the ground truth (labelsTs/).
    3. Calculates Dice/IoU scores for each case.
    4. Produces side-by-side visuals for each case: (MRI slice, prediction, GT).
       Saves them as .png in an output folder for easy review.

**Assumptions**:
    - We have /scratch/09999/aymanmahfuz/pancreas_eval_data_preprocessed/
      ├─ imagesTs/ (CaseID_0000.nii.gz)
      └─ labelsTs/ (CaseID.nii.gz)    # ground truth
    - We have a best model under e.g. Task333_T2PancreasMega.
    - We'll do single-slice visuals for each case (just a center slice, for instance).
      For deeper visualization, adapt code.

**Usage**:
    python evaluate_inference.py
"""

import os
import subprocess
import SimpleITK as sitk
import numpy as np
import matplotlib
matplotlib.use('Agg')  # for non-interactive HPC environment
import matplotlib.pyplot as plt

EVAL_BASE = "/scratch/09999/aymanmahfuz/pancreas_eval_data_preprocessed"
IMAGES_TS_DIR = os.path.join(EVAL_BASE, "imagesTs")
LABELS_TS_DIR = os.path.join(EVAL_BASE, "labelsTs")

# Prediction output directory
PRED_DIR = os.path.join(EVAL_BASE, "inference_predictions")

# Path to nnU-Net model info
TASK_ID = 333
TASK_NAME = "Task333_T2PancreasMega"

TRAINER = "nnTransUNetTrainerV2"
PLANS = "nnUNetPlansv2.1"
CONFIG = "3d_fullres"

# We assume you want to do single fold or the default ensemble.
# If single fold, specify --folds 0 or similar. If not, will use all folds.

def run_inference():
    """
    Calls nnUNet_predict using subprocess. Writes predictions to PRED_DIR.
    """
    os.makedirs(PRED_DIR, exist_ok=True)

    cmd = [
        "nnUNet_predict",
        "-i", IMAGES_TS_DIR,
        "-o", PRED_DIR,
        "-tr", TRAINER,
        "-m", CONFIG,
        "-p", PLANS,
        "-t", TASK_NAME,
        #"--folds", "0"  # Uncomment if you want only fold 0, for example
    ]

    print("Running command:", " ".join(cmd))
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error running inference:")
        print(result.stderr)
        raise RuntimeError("nnUNet_predict failed")
    else:
        print("Inference completed successfully!")
        print(result.stdout)

def dice_coefficient(pred, gt, label=1):
    """
    pred, gt: NumPy arrays of the same shape
    label: integer label for the structure of interest (pancreas=1)
    Returns Dice for that label.
    """
    pred_bin = (pred == label)
    gt_bin   = (gt == label)
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    denom = pred_bin.sum() + gt_bin.sum()
    dice = 2.0 * intersection / (denom + 1e-8)
    return dice

def iou_coefficient(pred, gt, label=1):
    """
    Intersection over Union for a given label.
    """
    pred_bin = (pred == label)
    gt_bin   = (gt == label)
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = pred_bin.sum() + gt_bin.sum() - intersection
    iou = intersection / (union + 1e-8)
    return iou

def visualize_case(case_id, out_png):
    """
    Creates a 3-panel figure: (mid slice of MRI, pred, gt).
    """
    # Read volumes
    mri_path  = os.path.join(IMAGES_TS_DIR, f"{case_id}_0000.nii.gz")
    pred_path = os.path.join(PRED_DIR, f"{case_id}.nii.gz")
    gt_path   = os.path.join(LABELS_TS_DIR, f"{case_id}.nii.gz")

    if not all(os.path.exists(x) for x in [mri_path, pred_path, gt_path]):
        print(f"[SKIP VIS] Missing files for {case_id}")
        return

    mri_sitk = sitk.ReadImage(mri_path)
    pred_sitk = sitk.ReadImage(pred_path)
    gt_sitk = sitk.ReadImage(gt_path)

    mri_arr = sitk.GetArrayFromImage(mri_sitk)  # shape (Z, Y, X)
    pred_arr = sitk.GetArrayFromImage(pred_sitk)
    gt_arr = sitk.GetArrayFromImage(gt_sitk)

    # We'll pick the middle slice in Z
    z_mid = mri_arr.shape[0] // 2
    mri_slice = mri_arr[z_mid, :, :]
    pred_slice = pred_arr[z_mid, :, :]
    gt_slice = gt_arr[z_mid, :, :]

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(mri_slice, cmap='gray')
    axs[0].set_title(f"{case_id} MRI (Z={z_mid})")
    axs[0].axis('off')

    axs[1].imshow(mri_slice, cmap='gray')
    axs[1].imshow(pred_slice, cmap='jet', alpha=0.4)
    axs[1].set_title("Prediction Overlay")
    axs[1].axis('off')

    axs[2].imshow(mri_slice, cmap='gray')
    axs[2].imshow(gt_slice, cmap='jet', alpha=0.4)
    axs[2].set_title("Ground Truth Overlay")
    axs[2].axis('off')

    plt.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

def main():
    # 1) Run Inference
    run_inference()

    # 2) Evaluate Dice/IoU for each case
    case_ids = []
    for f in os.listdir(IMAGES_TS_DIR):
        if f.endswith("_0000.nii.gz"):
            case_id = f.replace("_0000.nii.gz", "")
            case_ids.append(case_id)
    case_ids.sort()

    dice_scores = []
    iou_scores = []
    out_vis_dir = os.path.join(PRED_DIR, "visuals")
    os.makedirs(out_vis_dir, exist_ok=True)

    for c in case_ids:
        pred_path = os.path.join(PRED_DIR, f"{c}.nii.gz")
        gt_path   = os.path.join(LABELS_TS_DIR, f"{c}.nii.gz")

        if not (os.path.exists(pred_path) and os.path.exists(gt_path)):
            print(f"[SKIP] Missing prediction or GT for {c}")
            continue

        pred_sitk = sitk.ReadImage(pred_path)
        gt_sitk   = sitk.ReadImage(gt_path)

        pred_arr = sitk.GetArrayFromImage(pred_sitk)
        gt_arr   = sitk.GetArrayFromImage(gt_sitk)

        d = dice_coefficient(pred_arr, gt_arr, label=1)
        i = iou_coefficient(pred_arr, gt_arr, label=1)
        dice_scores.append(d)
        iou_scores.append(i)

        print(f"Case {c}: Dice={d:.4f}, IoU={i:.4f}")

        # 3) Generate side-by-side visuals
        out_png = os.path.join(out_vis_dir, f"{c}_visual.png")
        visualize_case(c, out_png)

    if len(dice_scores) > 0:
        avg_dice = np.mean(dice_scores)
        avg_iou  = np.mean(iou_scores)
        print(f"\n=== Summary over {len(dice_scores)} cases ===")
        print(f"Avg Dice: {avg_dice:.4f}")
        print(f"Avg IoU : {avg_iou:.4f}")
    else:
        print("No predictions were evaluated. Check naming or paths.")

if __name__ == "__main__":
    main()
