""" function for training and validation in one epoch
    Yunli Qi
"""

import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from monai.losses import DiceLoss, FocalLoss
from tqdm import tqdm

import cfg
from conf import settings
from func_3d.utils import eval_seg

# Remove global args usage and GPUdevice setup here
# Remove global pos_weight, criterion_G, and paper_loss here

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

# Avoid defining GPU or losses globally
torch.backends.cudnn.benchmark = True
scaler = torch.cuda.amp.GradScaler()
max_iterations = settings.EPOCH
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []

def train_sam(args, net: nn.Module, optimizer1, optimizer2, train_loader, epoch):
    # Define GPUdevice and loss inside function
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    pos_weight = torch.ones([1]).cuda(device=GPUdevice)*2
    criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # If needed, define paper_loss here if you actually use it
    paper_loss = CombinedLoss(dice_weight=1/21, focal_weight=20/21)

    net.train()
    if optimizer1 is not None:
        optimizer1.zero_grad()
    if optimizer2 is not None:
        optimizer2.zero_grad()

    video_length = args.video_length
    prompt = args.prompt
    prompt_freq = args.prompt_freq
    lossfunc = criterion_G

    epoch_loss = 0
    epoch_prompt_loss = 0
    epoch_non_prompt_loss = 0

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for pack in train_loader:
            torch.cuda.empty_cache()
            imgs_tensor = pack['image']
            mask_dict = pack['label']
            if prompt == 'click':
                pt_dict = pack['pt']
                point_labels_dict = pack['p_label']
            elif prompt == 'bbox':
                bbox_dict = pack['bbox']
            imgs_tensor = imgs_tensor.squeeze(0).to(dtype=torch.float32, device=GPUdevice)
            
            train_state = net.train_init_state(imgs_tensor=imgs_tensor)
            prompt_frame_id = list(range(0, video_length, prompt_freq))
            obj_list = []
            for id in prompt_frame_id:
                obj_list += list(mask_dict[id].keys())
            obj_list = list(set(obj_list))
            if len(obj_list) == 0:
                continue

            name = pack['image_meta_dict']['filename_or_obj']

            with torch.cuda.amp.autocast():
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
                for out_frame_idx, out_obj_ids, out_mask_logits in net.train_propagate_in_video(train_state, start_frame_idx=0):
                    video_segments[out_frame_idx] = {
                        out_obj_id: out_mask_logits[i]
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }

                loss = 0
                non_prompt_loss = 0
                prompt_loss = 0
                for id in range(video_length):
                    for ann_obj_id in obj_list:
                        pred = video_segments[id][ann_obj_id].unsqueeze(0)
                        try:
                            mask = mask_dict[id][ann_obj_id].to(dtype=torch.float32, device=GPUdevice)
                        except KeyError:
                            mask = torch.zeros_like(pred).to(device=GPUdevice)
                        obj_loss = lossfunc(pred, mask)
                        loss += obj_loss.item()
                        if id in prompt_frame_id:
                            prompt_loss += obj_loss
                        else:
                            non_prompt_loss += obj_loss

                loss = loss / video_length / len(obj_list)
                if prompt_freq > 1:
                    non_prompt_loss = non_prompt_loss / (video_length - len(prompt_frame_id)) / len(obj_list)
                prompt_loss = prompt_loss / len(prompt_frame_id) / len(obj_list)

                pbar.set_postfix(**{'loss (batch)': loss})
                epoch_loss += loss
                epoch_prompt_loss += prompt_loss.item()
                if prompt_freq > 1:
                    epoch_non_prompt_loss += non_prompt_loss.item()

                if (non_prompt_loss is not int) and (optimizer2 is not None) and prompt_freq > 1:
                    non_prompt_loss.backward(retain_graph=True)
                    optimizer2.step()
                if optimizer1 is not None:
                    prompt_loss.backward()
                    optimizer1.step()
                    optimizer1.zero_grad()
                if optimizer2 is not None:
                    optimizer2.zero_grad()

                net.reset_state(train_state)

            pbar.update()

    return epoch_loss / len(train_loader), epoch_prompt_loss / len(train_loader), epoch_non_prompt_loss / len(train_loader)


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
    mask_tensor: torch.Tensor = None,  # (optional) for measuring IoU/DICE
    bbox_dict: dict = None,           # bounding-box prompts: bbox_dict[frame_idx][obj_id] = Tensor(...)
    prompt: str = "bbox",
    prompt_freq: int = 2,
    device_str: str = "cuda",
    visualize: bool = False,
    out_dir: str = "./infer_outputs",
    case_name: str = "case0",
    threshold_list = (0.5,),  # or (0.1,0.3,0.5,0.7,0.9) if you want multi-threshold
):
    """
    Perform multi-frame (3D) inference on 'imgs_tensor' using the existing 'add_new_bbox'
    and 'propagate_in_video' calls from SAM2VideoPredictor.

    net:
      - Should be an instance of SAM2VideoPredictor in eval mode
      - Must have the methods: val_init_state(...), add_new_bbox(...), propagate_in_video(...)

    imgs_tensor:
      - shape [T, 3, H, W] or [T, C, H, W], in float32
      - entire volume, not just 2 frames

    mask_tensor:
      - optional ground truth [T, 1, H, W], for computing IoU/Dice

    bbox_dict:
      - bounding-box dictionary: e.g., bbox_dict[t][0] = bounding_box_tensor
      - bounding_box_tensor shape [1,4]: (x_min, y_min, x_max, y_max)

    prompt:
      - "bbox" or "click" (here we demonstrate "bbox")

    prompt_freq:
      - how often to add prompts (e.g. every 2 frames)

    device_str:
      - "cuda" or "cpu"

    visualize:
      - whether to save per-frame predictions as PNG

    out_dir, case_name:
      - where to save the optional PNG results

    threshold_list:
      - thresholds used to measure IoU/Dice if mask_tensor is provided
      - commonly (0.5,) or multiple thresholds

    Returns:
      A dict with:
        "prediction_3d": shape [T,1,H,W] of raw predicted masks (after sigmoid)
        "mean_iou":  averaged IoU across frames (if mask_tensor is provided)
        "mean_dice": averaged Dice across frames (if mask_tensor is provided)
    """

    import os
    import torch
    import matplotlib.pyplot as plt
    import numpy as np

    device = torch.device(device_str)
    net.eval()

    os.makedirs(out_dir, exist_ok=True)

    # 1) Move the entire volume to device
    imgs_tensor = imgs_tensor.to(device=device, dtype=torch.float32)

    # Move optional GT mask to device
    if mask_tensor is not None:
        mask_tensor = mask_tensor.to(device=device, dtype=torch.float32)

    # 2) Initialize inference state in val-mode
    inference_state = net.val_init_state(imgs_tensor=imgs_tensor)

    num_frames = imgs_tensor.shape[0]

    # 3) Add prompts (bounding boxes) for frames at the given prompt frequency
    #    but we must call net.add_new_bbox, not val_add_new_bbox
    prompt_frames = range(0, num_frames, prompt_freq)

    if bbox_dict is not None and prompt == "bbox":
        for f_idx in prompt_frames:
            if f_idx in bbox_dict:
                for obj_id, bbox_tsr in bbox_dict[f_idx].items():
                    # This calls the existing inference-mode method `add_new_bbox`
                    # in sam2_video_predictor.py
                    net.add_new_bbox(
                        inference_state=inference_state,
                        frame_idx=f_idx,
                        obj_id=obj_id,
                        bbox=bbox_tsr,
                        clear_old_points=False,  # or True, up to you
                        normalize_coords=True
                    )

    elif bbox_dict is not None and prompt == "click":
        # If you had clicks, you'd do net.add_new_points(...) here
        # but we won't detail that
        pass

    else:
        # No prompts => purely forward pass. Then we won't call add_new_bbox
        pass

    # 4) Propagate in video from frame=0 to the end
    #    This yields a generator of (frame_idx, obj_ids, mask_logits) per frame
    predicted_logits_dict = {}  # store raw logits for each frame
    with torch.no_grad():
        for frame_idx, obj_ids, mask_logits in net.propagate_in_video(
            inference_state=inference_state,
            start_frame_idx=0,
            max_frame_num_to_track=num_frames,  # track all frames
            reverse=False,
        ):
            # mask_logits shape => [num_objects, H, W]
            # By default we have only 1 object => shape [1,H,W]
            # Let's store them
            predicted_logits_dict[frame_idx] = mask_logits.detach().clone()

    # 5) Gather the final predictions in ascending frame index
    predicted_3d_list = []
    for t in range(num_frames):
        if t not in predicted_logits_dict:
            # If we never wrote anything, produce zeros
            shape_2d = imgs_tensor.shape[2:]  # (H,W)
            predicted_3d_list.append(
                torch.zeros((1, *shape_2d), device=device, dtype=torch.float32)
            )
        else:
            predicted_3d_list.append(predicted_logits_dict[t])

    # Combine into shape [T,1,H,W]
    predicted_3d = torch.stack(predicted_3d_list, dim=0)  # [T,1,H,W]

    # Convert from logits => probabilities
    predicted_3d = torch.sigmoid(predicted_3d)

    # 6) Evaluate IoU / Dice if we have GT
    iou_accum = 0.0
    dice_accum = 0.0
    count_slices = 0

    if mask_tensor is not None:
        for t in range(num_frames):
            pred_slice = predicted_3d[t]   # shape [1,H,W]
            gt_slice   = mask_tensor[t]    # shape [1,H,W]
            # Evaluate for each threshold in threshold_list
            for th in threshold_list:
                pred_bin = (pred_slice > th).float()
                gt_bin   = (gt_slice   > th).float()
                intersect = (pred_bin * gt_bin).sum()
                union     = (pred_bin + gt_bin - pred_bin*gt_bin).sum() + 1e-6
                iou_val   = (intersect / union).item()
                dice_val  = (2.0 * intersect / (pred_bin.sum()+gt_bin.sum()+1e-6)).item()

                # If we only accumulate at threshold=0.5, do:
                if abs(th-0.5)<1e-3:
                    iou_accum  += iou_val
                    dice_accum += dice_val
                    count_slices += 1

    mean_iou  = (iou_accum / count_slices) if (count_slices>0) else 0.0
    mean_dice = (dice_accum / count_slices) if (count_slices>0) else 0.0

    # 7) Optional visualization
    if visualize:
        import numpy as np
        import os
        os.makedirs(os.path.join(out_dir, case_name), exist_ok=True)

        for t in range(num_frames):
            # shape [C,H,W] => for the MRI
            # or if your input is [3,H,W] => do something like:
            input_np = imgs_tensor[t].cpu().numpy()  # [3,H,W]
            # shape [H,W]
            pred_np  = (predicted_3d[t].cpu().numpy()[0] > 0.5).astype(np.uint8)

            fig, axes = plt.subplots(1,2, figsize=(8,4))
            axes[0].imshow( np.transpose(input_np, (1,2,0)) )  # (H,W,3)
            axes[0].set_title(f"Input slice {t}")
            axes[0].axis("off")

            axes[1].imshow(pred_np, cmap="jet")
            axes[1].set_title("Prediction >0.5")
            axes[1].axis("off")

            save_path = os.path.join(out_dir, case_name, f"frame_{t:03d}.png")
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()

    return {
        "prediction_3d": predicted_3d,  # shape [T,1,H,W]
        "mean_iou":  mean_iou,
        "mean_dice": mean_dice,
    }
