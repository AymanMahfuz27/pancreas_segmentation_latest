import os
import sys
import json
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil

sys.path.append('.')  # ensure we can import from PROJECT_DIR if needed
from func_3d.function import infer_single_video

def dice_coeff_binary(pred, gt):
    """Dice for 2D with threshold=0.5."""
    intersection = np.sum((pred>0.5) & (gt>0.5))
    union = np.sum(pred>0.5) + np.sum(gt>0.5)
    if union == 0:
        return 1.0 if (np.sum(gt)==0 and np.sum(pred)==0) else 0.0
    return 2.0 * intersection / union

def iou_binary(pred, gt):
    """IOU for 2D with threshold=0.5."""
    intersection = np.sum((pred>0.5) & (gt>0.5))
    total = np.sum((pred>0.5) | (gt>0.5))
    if total == 0:
        return 1.0 if (np.sum(gt)==0 and np.sum(pred)==0) else 0.0
    return intersection / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Build minimal arguments for get_network
    class TrainArgs:
        pass
    train_args = TrainArgs()
    train_args.gpu = True
    train_args.gpu_device = 0
    train_args.net = "sam2"
    # Use the checkpoint path inside container
    train_args.sam_ckpt = "/checkpoints/sam2_hiera_small.pt"
    train_args.sam_config = "sam2_hiera_s"
    train_args.image_size = 1024
    train_args.prompt = "bbox"
    train_args.distributed = "none"
    train_args.multimask_output = 0
    train_args.memory_bank_size = 32

    device = torch.device("cuda:0")
    from func_3d.utils import get_network

    # Build the SAM2VideoPredictor model, load your "base" checkpoint, then load your best_model:
    net = get_network(train_args, train_args.net, use_gpu=True, gpu_device=device, distribution='none')
    ckpt = torch.load(args.model_path, map_location=device)
    net.load_state_dict(ckpt['model'], strict=True)
    net.eval().to(device=device, dtype=torch.float32)

    # 2) Gather all case directories in data_dir
    data_dir = Path(args.data_dir)
    case_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])

    dice_all = []
    iou_all  = []

    import torchvision.transforms.functional as F
    from PIL import Image

    def min_max_norm(img):
        mn, mx = img.min(), img.max()
        if mx>mn:
            return (img - mn)/(mx - mn)
        else:
            return img*0

    for case_dir in tqdm(case_dirs, desc="Evaluating Cases"):
        vol_path  = case_dir / "volume.nii.gz"
        mask_path = case_dir / "mask.nii.gz"
        if not (vol_path.exists() and mask_path.exists()):
            print(f"Skipping {case_dir.name}: Missing volume/mask .nii.gz.")
            continue

        # 3) Load volume & mask via nib
        vol_nib  = nib.load(str(vol_path))
        mask_nib = nib.load(str(mask_path))
        vol_data = nib.load(str(vol_path)).get_fdata().astype(np.float32)
        print(f"[BEFORE any fix] vol_data.shape = {vol_data.shape}")
        mask_data = nib.load(str(mask_path)).get_fdata().astype(np.float32)
        print(f"[BEFORE any fix] mask_data.shape = {mask_data.shape}")
        mask_data = (mask_data>0).astype(np.float32)
        vol_data = np.transpose(vol_data, (2, 0, 1))   # shape now (48, 1024, 1024)
        mask_data= np.transpose(mask_data,(2, 0, 1))   # shape now (48, 1024, 1024)

        Z,H,W = vol_data.shape
        print(f"[AFTER fix] vol_data.shape = {vol_data.shape}")   # (48, 1024, 1024)
        print(f"[AFTER fix] mask_data.shape = {mask_data.shape}")

        # Preprocess to shape [Z,3,1024,1024]
        mri_slices_list = []
        mask_slices_list = []

        for z in range(Z):
            slice_2d = vol_data[z,...]
            slice_norm = min_max_norm(slice_2d)
            slice_pil = Image.fromarray((slice_norm*255).astype(np.uint8))
            slice_pil = F.resize(slice_pil, (train_args.image_size, train_args.image_size))
            slice_arr = np.array(slice_pil, dtype=np.float32)/255.0
            # replicate to 3 channels
            mri_slices_list.append(np.stack([slice_arr]*3, axis=0))

            # mask
            mask_2d = mask_data[z,...]
            mask_pil = Image.fromarray((mask_2d*255).astype(np.uint8))
            mask_pil = F.resize(mask_pil, (train_args.image_size, train_args.image_size),
                                interpolation=Image.NEAREST)
            mask_arr = np.array(mask_pil, dtype=np.float32)/255.0
            mask_slices_list.append(mask_arr)

        mri_slices = np.stack(mri_slices_list, axis=0)   # [Z,3,1024,1024]
        mask_slices= np.stack(mask_slices_list, axis=0) # [Z,1024,1024]

        # build a bounding box for each slice
        bbox_dict = {}
        for z in range(Z):
            mslice = mask_slices[z]
            print(f"Slice {z}, mask sum = {mslice.sum()}")
            nz = np.argwhere(mslice > 0.5)   # shape => (N, 2), meaning (row, col)

            if nz.size > 0:
                min_row, min_col = nz.min(axis=0)
                max_row, max_col = nz.max(axis=0)
            else:
                min_row, min_col = 0, 0
                max_row, max_col = mslice.shape[0] - 1, mslice.shape[1] - 1

            # Since your model needs [x_min, y_min, x_max, y_max],
            # we treat `col` as `x` and `row` as `y`.
            x_min, y_min = min_col, min_row
            x_max, y_max = max_col, max_row

            bbox = torch.tensor([x_min, y_min, x_max, y_max], 
                                device=device, dtype=torch.float32).unsqueeze(0)
            bbox_dict[z] = {0: bbox}


        # 4) Inference => shape [Z,1,1024,1024]
        imgs_tensor = torch.from_numpy(mri_slices).to(device=device, dtype=torch.float32)
        results = infer_single_video(
            net=net,
            imgs_tensor=imgs_tensor,
            mask_tensor=None,
            bbox_dict=bbox_dict,
            prompt="bbox",
            prompt_freq=1,     # bounding box each slice
            device_str="cuda",
            visualize=True,
            out_dir=args.output_dir,
            case_name=str(case_dir.name),
        )
        pred_4d = results["prediction_3d"]  # [Z,1,1024,1024]

        # measure dice, iou
        dice_case = []
        iou_case  = []
        pred_4d_np = pred_4d.cpu().numpy()

        for z in range(Z):
            pred_slice = pred_4d_np[z,0]   # shape [1024,1024]
            gt_slice   = mask_slices[z]    # shape [1024,1024]
            dval = dice_coeff_binary(pred_slice, gt_slice)
            ival = iou_binary(pred_slice, gt_slice)
            dice_case.append(dval)
            iou_case.append(ival)

        dice_mean = float(np.mean(dice_case))
        iou_mean  = float(np.mean(iou_case))
        dice_all.append(dice_mean)
        iou_all.append(iou_mean)
        print(f"[{case_dir.name}] => DICE={dice_mean:.4f}, IOU={iou_mean:.4f}")

    # 5) Summaries
    dice_final = float(np.mean(dice_all)) if dice_all else 0.0
    iou_final  = float(np.mean(iou_all))  if iou_all else 0.0
    summary = {
        "average_dice": dice_final,
        "average_iou":  iou_final,
        "num_cases": len(dice_all)
    }
    with open(os.path.join(args.output_dir, "evaluation_results.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print("=== Done evaluating on preprocessed NIfTI data ===")
    print(f"Overall: Dice={dice_final:.4f}, IoU={iou_final:.4f}")

if __name__ == "__main__":
    main()
