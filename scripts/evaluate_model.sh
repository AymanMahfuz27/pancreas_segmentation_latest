#!/bin/bash
#SBATCH -J eval_new_data
#SBATCH -o eval_new_data.o%j
#SBATCH -e eval_new_data.e%j
#SBATCH -p gpu-h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aymanmahfuz27@utexas.edu

PROJECT_DIR=/home1/09999/aymanmahfuz/pancreas_project_latest/pancreas_segmentation_latest
WORK_DIR=/work/09999/aymanmahfuz/ls6
SCRATCH_DIR=/scratch/09999/aymanmahfuz
CONTAINER_PATH=$WORK_DIR/containers/medsam2.sif

module purge
module load tacc-apptainer

cd "$PROJECT_DIR"

# Path to your best model
BEST_MODEL_PATH="$PROJECT_DIR/logs/pancreas_full_training_2024_12_20_01_36_35/Model/best_model.pth"

# Evaluation data directory (contains subdirs each with mri.dcm, mask.dcm)
EVAL_DATA_DIR="/scratch/pancreas_eval_data"

# Re-create the output directory
rm -rf "$SCRATCH_DIR/eval_results"
mkdir -p "$SCRATCH_DIR/eval_results"
OUTPUT_DIR="/scratch/eval_results"

#####################################
# Create new run_evaluation_on_new_data.py
#####################################
cat << 'EOF' > run_evaluation_on_new_data.py
import os
import torch
import numpy as np
import pydicom
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms.functional import resize
import json
from tqdm import tqdm
import argparse

class TrainArgs:
    pass

def normalize_and_resize_slice(slice_array, target_size=1024):
    """Normalize the slice to [0,1], then resize to target_size."""
    if slice_array.max() > 0:
        slice_array = slice_array / slice_array.max()
    slice_pil = Image.fromarray((slice_array * 255).astype(np.uint8))
    slice_pil = resize(slice_pil, (target_size, target_size))
    slice_resized = np.array(slice_pil, dtype=np.float32) / 255.0
    return slice_resized

def dice_coefficient(pred, gt):
    """Binary Dice on [H,W] arrays with threshold=0.5."""
    intersection = np.sum((pred > 0.5) & (gt > 0.5))
    union = np.sum(pred > 0.5) + np.sum(gt > 0.5)
    if union == 0:
        # Perfect if both are empty
        return 1.0 if (np.sum(gt)==0 and np.sum(pred)==0) else 0.0
    return 2.0 * intersection / union

def iou_score(pred, gt):
    """Binary IoU on [H,W] arrays with threshold=0.5."""
    intersection = np.sum((pred > 0.5) & (gt > 0.5))
    total = np.sum((pred > 0.5) | (gt > 0.5))
    if total == 0:
        return 1.0 if (np.sum(gt)==0 and np.sum(pred)==0) else 0.0
    return intersection / total

def create_overlay_image(mri_slice_3ch, gt_mask, pred_mask, output_path):
    """
    mri_slice_3ch: (C,H,W)= (3,H,W) in [0,1], final shape => (H,W,3)
    gt_mask, pred_mask: [H,W], binary
    """
    # Convert (3,H,W) -> (H,W,3)
    slice_hw3 = np.transpose(mri_slice_3ch, (1,2,0))
    base_img_rgb = (slice_hw3 * 255).astype(np.uint8)

    # Mark GT in green channel
    base_img_rgb[gt_mask > 0.5, 1] = 255
    # Mark prediction in red channel
    base_img_rgb[pred_mask > 0.5, 0] = 255

    Image.fromarray(base_img_rgb).save(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Build "train_args" needed by get_network
    train_args = TrainArgs()
    train_args.gpu = True
    train_args.gpu_device = 0
    train_args.net = 'sam2'
    train_args.sam_ckpt = '/work/checkpoints/sam2_hiera_small.pt'
    train_args.sam_config = 'sam2_hiera_t'
    train_args.image_size = 1024
    train_args.prompt = 'bbox'
    train_args.distributed = 'none'
    train_args.multimask_output = 0
    train_args.memory_bank_size = 32

    from func_3d.utils import get_network
    from func_3d.function import infer_single_video

    # Load the model
    device = torch.device('cuda', 0)
    net = get_network(train_args, train_args.net, use_gpu=True, gpu_device=device, distribution='none')
    checkpoint = torch.load(args.model_path, map_location=device)
    net.load_state_dict(checkpoint['model'], strict=True)
    net.eval()
    net.to(device=device, dtype=torch.float32)

    # Gather all case directories
    data_dir = Path(args.data_dir)
    cases = [d for d in data_dir.iterdir() if d.is_dir()]

    dice_scores_all = []
    iou_scores_all = []

    for case_dir in tqdm(cases, desc="Evaluating new cases"):
        mri_path = case_dir / 'mri.dcm'
        mask_path = case_dir / 'mask.dcm'
        if not (mri_path.exists() and mask_path.exists()):
            print(f"Skipping {case_dir.name} due to missing DICOM file.")
            continue

        # Load full volume
        mri_dcm = pydicom.dcmread(str(mri_path))
        mask_dcm = pydicom.dcmread(str(mask_path))
        mri_vol = mri_dcm.pixel_array.astype(np.float32)   # shape = (T, H, W)
        mask_vol = mask_dcm.pixel_array.astype(np.float32) # shape = (T, H, W)
        mask_vol = (mask_vol > 0).astype(np.float32)

        T_vol = mri_vol.shape[0]
        # We set "video_length" to T_vol so the model sees the entire volume
        train_args.video_length = T_vol

        # Preprocess all slices in the volume
        # We'll create mri_slices => shape [T,3,H,W]
        mri_slices_list = []
        mask_slices_list = []
        for i in range(T_vol):
            slice_2d = mri_vol[i]  # shape (H,W)
            slice_resized = normalize_and_resize_slice(slice_2d, target_size=train_args.image_size)
            # => shape (H,W) in [0,1]
            # stack 3 copies for [3,H,W]
            mri_slices_list.append(np.stack([slice_resized]*3, axis=0))

            # Also do the same for the mask
            mask_2d = mask_vol[i]  # shape (H,W)
            mask_pil = Image.fromarray((mask_2d*255).astype(np.uint8))
            mask_pil = resize(mask_pil, (train_args.image_size, train_args.image_size))
            mask_resized = np.array(mask_pil, dtype=np.float32)/255.0
            # shape => (H,W)
            mask_slices_list.append(mask_resized)

        # Convert lists to [T,3,H,W] and [T,H,W]
        mri_slices = np.stack(mri_slices_list, axis=0)   # => shape [T,3,H,W]
        mask_slices = np.stack(mask_slices_list, axis=0) # => shape [T,H,W]

        # Move MRI to torch
        mri_tensor = torch.from_numpy(mri_slices).to(device=device, dtype=torch.float32)

        # Build a bounding box for each frame from the GT mask
        bbox_dict = {}
        for t in range(T_vol):
            frame_mask = mask_slices[t]
            nz = np.argwhere(frame_mask > 0.5)
            if nz.size > 0:
                y_min, x_min = nz.min(axis=0)
                y_max, x_max = nz.max(axis=0)
            else:
                # If no pancreas in that slice, just do a minimal box
                y_min, x_min = 0, 0
                y_max, x_max = frame_mask.shape[0]-1, frame_mask.shape[1]-1
            bbox = torch.tensor([x_min,y_min,x_max,y_max],
                                dtype=torch.float32, device=device).unsqueeze(0)
            bbox_dict[t] = {0: bbox}

        # Run full-volume inference
        # We'll pass mask_tensor=None (since we only use bounding boxes for prompts)
        results = infer_single_video(
            net=net,
            imgs_tensor=mri_tensor,
            mask_tensor=None,   # we do not pass the GT here
            bbox_dict=bbox_dict,
            prompt="bbox",
            prompt_freq=1,      # prompt each frame if you want
            device_str="cuda",
            visualize=False,     # set True if you want saved PNGs
            out_dir=args.output_dir,
            case_name=case_dir.name
        )
        pred_4d = results["prediction_3d"]  # shape [T,1,H,W], raw logits from model

        # Evaluate Dice / IoU across ALL T_vol slices
        dice_list = []
        iou_list = []
        for t in range(T_vol):
            pred_mask_t = pred_4d[t,0].cpu().numpy()   # shape (H,W)
            gt_mask_t   = mask_slices[t]               # shape (H,W), float in [0,1]

            d = dice_coefficient(pred_mask_t, gt_mask_t)
            i = iou_score(pred_mask_t, gt_mask_t)
            dice_list.append(d)
            iou_list.append(i)

            # Optionally create an overlay for each slice
            # (This can be a lot of images if T_vol is large!)
            # Let's do it only for e.g. the center slice
            # or skip entirely to avoid clutter.
            """
            overlay_path = os.path.join(
                args.output_dir,
                f"{case_dir.name}_slice{t}.png"
            )
            create_overlay_image(
                mri_slices[t],    # [3,H,W]
                gt_mask_t,        # [H,W]
                pred_mask_t,      # [H,W]
                overlay_path
            )
            """

        case_dice = float(np.mean(dice_list))
        case_iou  = float(np.mean(iou_list))
        dice_scores_all.append(case_dice)
        iou_scores_all.append(case_iou)
        print(f"Case {case_dir.name}: Dice={case_dice:.4f}, IoU={case_iou:.4f}")

    # Summarize
    avg_dice = float(np.mean(dice_scores_all)) if dice_scores_all else 0.0
    avg_iou  = float(np.mean(iou_scores_all))  if iou_scores_all else 0.0

    summary_json = {
        "average_dice": avg_dice,
        "average_iou":  avg_iou,
        "num_cases":    len(cases)
    }

    summary_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(summary_path, 'w') as f:
        json.dump(summary_json, f, indent=4)

    print("\n=== Evaluation Complete ===")
    print(f"Average Dice: {avg_dice:.4f}")
    print(f"Average IoU:  {avg_iou:.4f}")
    print("Detailed results saved in evaluation_results.json")
EOF

#####################################
# Now run inside the container
#####################################
apptainer exec --nv \
  --bind "$SCRATCH_DIR:/scratch" \
  --bind "$PROJECT_DIR:/project" \
  --bind "$WORK_DIR:/work" \
  "$CONTAINER_PATH" \
  bash -c "
    source /opt/conda/etc/profile.d/conda.sh && conda activate medsam2 && \
    python run_evaluation_on_new_data.py \
      --model_path '$BEST_MODEL_PATH' \
      --data_dir '$EVAL_DATA_DIR' \
      --output_dir '$OUTPUT_DIR'
  "

echo "Evaluation completed. Check $SCRATCH_DIR/eval_results for results."
