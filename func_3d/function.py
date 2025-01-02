""" function for training and validation in one epoch
    Yunli Qi
"""
##############################################################################
# function.py
##############################################################################
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from monai.losses import DiceLoss, FocalLoss
from tqdm import tqdm
import sys

import cfg
from conf import settings
from func_3d.utils import eval_seg

class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=1, focal_weight=1):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss(to_onehot_y=True, sigmoid=True)
        self.focal_loss = FocalLoss(to_onehot_y=True, gamma=2.0)

    def forward(self, inputs, targets):
        dice = self.dice_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)
        return self.dice_weight * dice + self.focal_weight * focal

torch.backends.cudnn.benchmark = True
scaler = torch.cuda.amp.GradScaler()
max_iterations = settings.EPOCH
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []

def train_sam(args, net: nn.Module, optimizer1, optimizer2, train_loader, epoch):
    """
    During training, we skip empty slices (i.e., slices with no pancreas),
    and only prompt a subset of the non-empty slices to avoid confusion.
    """
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    pos_weight = torch.ones([1]).cuda(device=GPUdevice)*2
    criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    paper_loss = CombinedLoss(dice_weight=1/21, focal_weight=20/21)

    net.train()
    if optimizer1 is not None:
        optimizer1.zero_grad()
    if optimizer2 is not None:
        optimizer2.zero_grad()

    video_length = args.video_length      # e.g. 30 or 48 slices
    prompt = args.prompt                 # "bbox" or "click"
    prompt_freq = args.prompt_freq       # how often we prompt => e.g. 2
    # but we will skip empty slices, see below
    lossfunc = criterion_G

    epoch_loss = 0.0
    epoch_prompt_loss = 0.0
    epoch_non_prompt_loss = 0.0

    # Choose how many non-empty slices to actually prompt
    # e.g. we can keep prompt_freq as is, or define a new param:
    skip_factor = 2  # skip empty frames, then further skip to every 2nd or 3rd
    
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for pack in train_loader:
            pbar.update()
            torch.cuda.empty_cache()
            imgs_tensor = pack['image']      # shape [Z, 3, H, W]
            if imgs_tensor.dim() == 5 and imgs_tensor.size(0) == 1:
                imgs_tensor = imgs_tensor.squeeze(0)  # => [Z, 3, H, W]

            mask_dict   = pack['label']      # dict: mask_dict[slice_z][obj_id] => [H,W]

            # possibly also have pt_dict, p_label if prompt=="click"
            # or bbox_dict if prompt=="bbox"
            if prompt == 'click':
                pt_dict       = pack.get('pt', None)
                point_lbl_dict= pack.get('p_label', None)
            elif prompt == 'bbox':
                bbox_dict     = pack.get('bbox', None)

            imgs_tensor = imgs_tensor.to(dtype=torch.float32, device=GPUdevice)

            # 1) Build train_state
            train_state = net.train_init_state(imgs_tensor=imgs_tensor)

            # 2) Identify the slices that actually have the object
            #    (since in your dataset, mask_dict[z] = {0: slice_mask} if non-empty)
            #    We'll gather them into a list, then sub-sample them
            nonempty_slices = []
            Z = imgs_tensor.size(0)
            for z in range(Z):
                # if there's something in mask_dict[z], let's see if slice_mask is >0
                if z in mask_dict and len(mask_dict[z])>0:
                    # e.g. mask_dict[z] = {0: <some mask>}
                    slice_mask = mask_dict[z][0]  # [H,W] tensor
                    if slice_mask.sum() > 0:
                        nonempty_slices.append(z)
            

            # sub-sample them => e.g. pick every nth
            # you could do: nonempty_slices = nonempty_slices[:: prompt_freq]
            # but let's do skip_factor. Tweak as you like.
            chosen_slices = nonempty_slices[::skip_factor]

            if len(chosen_slices)==0:
                # no pancreas? skip
                continue

            # 3) We have only "obj_id=0" in your dataset. If you had multiple, collect them
            obj_list = [0]  # or gather from mask_dict if you have multiple objects

            # 4) Actually add prompts for only these chosen slices
            with torch.cuda.amp.autocast():
                for z_idx in chosen_slices:
                    for ann_obj_id in obj_list:
                        try:
                            if prompt=='bbox' and bbox_dict is not None:
                                bbox = bbox_dict[z_idx][ann_obj_id].to(device=GPUdevice)
                                net.train_add_new_bbox(train_state, z_idx, ann_obj_id, bbox, False)
                            elif prompt=='click' and pt_dict is not None:
                                # handle clicks
                                pass
                            else:
                                # fallback
                                net.train_add_new_mask(
                                    train_state, z_idx, ann_obj_id,
                                    torch.zeros(imgs_tensor.shape[2:]).to(GPUdevice)
                                )
                        except KeyError:
                            # if there's no bounding box for that slice
                            net.train_add_new_mask(
                                train_state, z_idx, ann_obj_id,
                                torch.zeros(imgs_tensor.shape[2:]).to(GPUdevice)
                            )

                # 5) Forward pass => get predicted masks on all frames
                video_segments = {}
                for out_frame_idx, out_obj_ids, out_mask_logits in net.train_propagate_in_video(
                    train_state, start_frame_idx=0):
                    video_segments[out_frame_idx] = {
                        o_id: out_mask_logits[i] for i, o_id in enumerate(out_obj_ids)
                    }

                # 6) Compute total loss across all slices
                loss_val = 0.0
                prompt_loss_val = 0.0
                non_prompt_loss_val = 0.0

                for z in range(Z):
                    for ann_obj_id in obj_list:
                        pred = video_segments[z][ann_obj_id].unsqueeze(0)  # => [1,H,W]
                        # get ground truth if available, else zero
                        if z in mask_dict and ann_obj_id in mask_dict[z]:
                            mask = mask_dict[z][ann_obj_id].to(dtype=torch.float32, device=GPUdevice)
                        else:
                            mask = torch.zeros_like(pred)

                        obj_loss = lossfunc(pred, mask)
                        loss_val += obj_loss.item()

                        if z in chosen_slices:
                            prompt_loss_val += obj_loss
                        else:
                            non_prompt_loss_val += obj_loss

                slice_count = video_length
                obj_count = len(obj_list)
                total_frames = slice_count * obj_count

                loss_val /= total_frames
                if len(chosen_slices) > 0:
                    prompt_loss_val /= (len(chosen_slices) * obj_count)
                if slice_count - len(chosen_slices) > 0:
                    non_prompt_loss_val /= ((slice_count - len(chosen_slices)) * obj_count)

                # accumulate epoch stats
                epoch_loss        += loss_val
                epoch_prompt_loss += prompt_loss_val.item()
                epoch_non_prompt_loss += non_prompt_loss_val.item()

                # 7) Backprop
                # in your original code, you had 2 optimizers => optimizer1, optimizer2
                # if you only have 1 => just call .backward() on the total
                # We'll do your existing logic:
                if (non_prompt_loss_val is not int) and optimizer2 is not None and len(chosen_slices)<slice_count:
                    non_prompt_loss_val.backward(retain_graph=True)
                    optimizer2.step()
                if optimizer1 is not None:
                    prompt_loss_val.backward()
                    optimizer1.step()
                    optimizer1.zero_grad()
                if optimizer2 is not None:
                    optimizer2.zero_grad()

                net.reset_state(train_state)

            # pbar.update()

    # Return averaged loss
    num_batches = len(train_loader)
    return (
        epoch_loss / num_batches,
        epoch_prompt_loss / num_batches,
        epoch_non_prompt_loss / num_batches
    )


def validation_sam(args, val_loader, epoch, net: nn.Module, clean_dir=True):
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    pos_weight = torch.ones([1]).cuda(device=GPUdevice)*2
    criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    net.eval()

    n_val = len(val_loader)
    mix_res = (0,)*2  # 2 for IOU and DICE
    tot = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    prompt_freq = args.prompt_freq
    prompt = args.prompt
    lossfunc = criterion_G

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for pack in val_loader:
            imgs_tensor = pack['image']
            mask_dict = pack['label']
            if prompt == 'click':
                pt_dict = pack['pt']
                point_labels_dict = pack['p_label']
            elif prompt == 'bbox':
                bbox_dict = pack['bbox']

            if len(imgs_tensor.size()) == 5:
                imgs_tensor = imgs_tensor.squeeze(0)
            imgs_tensor = imgs_tensor.to(dtype=torch.float32, device=GPUdevice)
            frame_id = list(range(imgs_tensor.size(0)))

            train_state = net.val_init_state(imgs_tensor=imgs_tensor)
            prompt_frame_id = list(range(0, len(frame_id), prompt_freq))
            obj_list = []
            for id in frame_id:
                obj_list += list(mask_dict[id].keys())
            obj_list = list(set(obj_list))
            if len(obj_list) == 0:
                pbar.update()
                continue

            name = pack['image_meta_dict']['filename_or_obj']

            with torch.no_grad():
                for id in prompt_frame_id:
                    for ann_obj_id in obj_list:
                        try:
                            if prompt == 'click':
                                points = pt_dict[id][ann_obj_id].to(device=GPUdevice)
                                labels = point_labels_dict[id][ann_obj_id].to(device=GPUdevice)
                                net.train_add_new_points(train_state, id, ann_obj_id, points, labels, False)
                            elif prompt == 'bbox':
                                bbox = bbox_dict[id][ann_obj_id].to(device=GPUdevice)
                                net.train_add_new_bbox(train_state, id, ann_obj_id, bbox, False)
                        except KeyError:
                            net.train_add_new_mask(train_state, id, ann_obj_id,
                                                   torch.zeros(imgs_tensor.shape[2:]).to(device=GPUdevice))

                video_segments = {}
                for out_frame_idx, out_obj_ids, out_mask_logits in net.propagate_in_video(train_state, start_frame_idx=0):
                    video_segments[out_frame_idx] = {
                        out_obj_id: out_mask_logits[i]
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }

                loss = 0
                pred_iou = 0
                pred_dice = 0
                for id in frame_id:
                    for ann_obj_id in obj_list:
                        pred = video_segments[id][ann_obj_id].unsqueeze(0)
                        try:
                            mask = mask_dict[id][ann_obj_id].to(dtype=torch.float32, device=GPUdevice)
                        except KeyError:
                            mask = torch.zeros_like(pred).to(device=GPUdevice)

                        loss += lossfunc(pred, mask)
                        temp = eval_seg(pred, mask, threshold)
                        pred_iou += temp[0]
                        pred_dice += temp[1]

                total_num = len(frame_id) * len(obj_list)
                loss = loss / total_num
                temp = (pred_iou / total_num, pred_dice / total_num)
                tot += loss
                mix_res = tuple([sum(a) for a in zip(mix_res, temp)])

            net.reset_state(train_state)
            pbar.update()

    return tot/n_val, tuple([a/n_val for a in mix_res])

def infer_single_video(
    net: nn.Module,
    imgs_tensor: torch.Tensor,
    mask_tensor: torch.Tensor = None,
    bbox_dict: dict = None,
    prompt: str = "bbox",
    prompt_freq: int = 2,
    device_str: str = "cuda",
    visualize: bool = False,
    out_dir: str = "./infer_outputs",
    case_name: str = "case0",
    threshold_list=(0.5,),
):
    """
    Multi-frame inference with optional bounding box prompts
    """

    import os
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    

    device = torch.device(device_str)
    net.eval()

    os.makedirs(out_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1) Move the entire volume to device, verify shape
    # -------------------------------------------------------------------------

    imgs_tensor = imgs_tensor.to(device=device, dtype=torch.float32)
    print(f"[DEBUG] Inference input shape = {tuple(imgs_tensor.shape)} "
          f"(should be [T,3,H,W] with T ~ #slices)")
    print(f"[DEBUG] Input shape: {imgs_tensor.shape}")
    print(f"[DEBUG] Volume stats - min: {imgs_tensor.min():.3f}, max: {imgs_tensor.max():.3f}, mean: {imgs_tensor.mean():.3f}")
    if mask_tensor is not None:
        print(f"[DEBUG] Mask stats - unique values: {torch.unique(mask_tensor).tolist()}")


    # e.g. for a 48-slice volume => (48, 3, 1024, 1024)

    if mask_tensor is not None:
        mask_tensor = mask_tensor.to(device=device, dtype=torch.float32)

    # 2) Initialize inference state
    inference_state = net.val_init_state(imgs_tensor=imgs_tensor)

    num_frames = imgs_tensor.shape[0]

    print(f"[DEBUG] num_frames = {num_frames}")
    if num_frames > 200:  # Just a soft check. If you see 1024 here, something’s off
        print("[WARN] Very large T dimension. Did you accidentally transpose?")

    # 3) Add bounding-box prompts
    prompt_frames = range(0, num_frames, prompt_freq)
    print(f"[DEBUG] Prompt frames: {list(prompt_frames)}")

    if bbox_dict is not None and prompt == "bbox":

        print(f"[DEBUG] Found bboxes for frames: {sorted(bbox_dict.keys())}")
        example_frame = next(iter(bbox_dict.keys()))
        print(f"[DEBUG] Example bbox for frame {example_frame}: {bbox_dict[example_frame]}")
        slices_with_box = []
        for z in sorted(bbox_dict.keys()):
            # check if there's a non-empty box
            if len(bbox_dict[z])>0:
                slices_with_box.append(z)
        # sub-sample if desired
        prompt_frames = slices_with_box[::2]  # every 2nd


        for f_idx in prompt_frames:
            if f_idx in bbox_dict:
                for obj_id, bbox_tsr in bbox_dict[f_idx].items():
                    # shape => [1,4], i.e. [x_min,y_min,x_max,y_max]
                    box = bbox_tsr[0].tolist()  # [x_min, y_min, x_max, y_max]
                    x_min, y_min, x_max, y_max = box

                    # quick sanity check:
                    if x_max <= x_min or y_max <= y_min:
                        print(f"[WARN] Invalid BBox at frame={f_idx} obj_id={obj_id}: "
                              f"x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")

                    net.add_new_bbox(
                        inference_state=inference_state,
                        frame_idx=f_idx,
                        obj_id=obj_id,
                        bbox=bbox_tsr,
                        clear_old_points=False,
                        normalize_coords=True,
                    )

    elif bbox_dict is not None and prompt == "click":
        # (unchanged)
        pass

    # 4) Propagate in video from frame=0
    predicted_logits_dict = {}
    with torch.no_grad():
        for frame_idx, obj_ids, mask_logits in net.propagate_in_video(
            inference_state=inference_state,
            start_frame_idx=0,
            max_frame_num_to_track=num_frames,
            reverse=False,
        ):
            predicted_logits_dict[frame_idx] = mask_logits.detach().clone()

    # 5) Gather final predictions in ascending frame index
    predicted_3d_list = []
    for t in range(num_frames):
        if t not in predicted_logits_dict:
            # If the model never wrote anything for this frame => zeros
            h, w = imgs_tensor.shape[2], imgs_tensor.shape[3]
            predicted_3d_list.append(
                torch.zeros((1, h, w), device=device, dtype=torch.float32)
            )
        else:
            predicted_3d_list.append(predicted_logits_dict[t])

    # shape => [T,1,H,W]
    predicted_3d = torch.stack(predicted_3d_list, dim=0)
    predicted_3d = torch.sigmoid(predicted_3d)

    # 6) Evaluate IoU / Dice if ground truth is available
    iou_accum = 0.0
    dice_accum = 0.0
    count_slices = 0
    if mask_tensor is not None:
        for t in range(num_frames):
            pred_slice = predicted_3d[t]   # shape => [1,H,W]
            gt_slice = mask_tensor[t]      # shape => [1,H,W]
            for th in threshold_list:
                pred_bin = (pred_slice > th).float()
                gt_bin = (gt_slice > th).float()
                intersect = (pred_bin * gt_bin).sum()
                union = (pred_bin + gt_bin - pred_bin * gt_bin).sum() + 1e-6
                iou_val = float(intersect / union)
                dice_val = float(2.0 * intersect / (pred_bin.sum() + gt_bin.sum() + 1e-6))
                if abs(th - 0.5) < 1e-3:
                    iou_accum += iou_val
                    dice_accum += dice_val
                    count_slices += 1

    mean_iou = iou_accum / count_slices if count_slices > 0 else 0.0
    mean_dice = dice_accum / count_slices if count_slices > 0 else 0.0

    # 7) Optional visualization
    if visualize:
        os.makedirs(os.path.join(out_dir, case_name), exist_ok=True)
        for t in range(num_frames):
            # shape => [3,H,W]
            input_np = imgs_tensor[t].cpu().numpy()

            # shape => [1,H,W] => squeeze => [H,W]
            pred_slice = predicted_3d[t].squeeze().cpu().numpy()
            pred_slice = (pred_slice > 0.5).astype(np.uint8)

            fig, axes = plt.subplots(1, 2, figsize=(9, 4))
            axes[0].imshow(np.transpose(input_np, (1, 2, 0)), cmap="gray")
            axes[0].set_title(f"Input slice {t}")
            axes[0].axis("off")

            axes[1].imshow(pred_slice, cmap="jet")
            axes[1].set_title(f"Pred (th=0.5), slice={t}")
            axes[1].axis("off")

            save_path = os.path.join(out_dir, case_name, f"frame_{t:03d}.png")
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()

    return {
        "prediction_3d": predicted_3d,
        "mean_iou": mean_iou,
        "mean_dice": mean_dice,
    }
