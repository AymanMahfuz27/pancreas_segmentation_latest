"""
Utility functions for training and evaluation.
    Yunli Qi
"""

import logging
import os
import random
import sys
import time
from datetime import datetime

import dateutil.tz
import numpy as np
import torch
from torch.autograd import Function

# No global cfg.parse_args() call here!
# import cfg  # If you need to import cfg, you can, but don't parse args at global level.

def get_network(args, net, use_gpu=True, gpu_device=0, distribution=True):
    """Return given network based on arguments."""
    device = torch.device('cuda', args.gpu_device)

    if net == 'sam2':
        from sam2_train.build_sam import build_sam2_video_predictor
        sam2_checkpoint = args.sam_ckpt
        model_cfg = args.sam_config
        net = build_sam2_video_predictor(
            config_file=model_cfg, 
            ckpt_path=sam2_checkpoint, 
            mode=None
        )
    else:
        print('The network name you have entered is not supported yet.')
        sys.exit()

    if use_gpu:
        net = net.to(device=gpu_device)
    return net


def create_logger(log_dir, phase='train'):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(time_str, phase)
    final_log_file = os.path.join(log_dir, log_file)
    head = '%(asctime)-15s %(message)s'

    logging.basicConfig(
        filename=str(final_log_file),
        format=head
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)
    return logger


def set_log_dir(root_dir, exp_name):
    """
    Create a directory structure to hold logs, checkpoints, etc.
    """
    path_dict = {}
    os.makedirs(root_dir, exist_ok=True)

    exp_path = os.path.join(root_dir, exp_name)
    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    prefix = f"{exp_path}_{timestamp}"
    os.makedirs(prefix)
    path_dict['prefix'] = prefix

    # Checkpoint directory
    ckpt_path = os.path.join(prefix, 'Model')
    os.makedirs(ckpt_path)
    path_dict['ckpt_path'] = ckpt_path

    # Log directory
    log_path = os.path.join(prefix, 'Log')
    os.makedirs(log_path)
    path_dict['log_path'] = log_path

    # Sample image directory
    sample_path = os.path.join(prefix, 'Samples')
    os.makedirs(sample_path)
    path_dict['sample_path'] = sample_path

    return path_dict


def save_checkpoint(states, is_best, output_dir, filename='checkpoint.pth'):
    """
    Save model checkpoint to output_dir.
    If is_best, also copy to 'checkpoint_best.pth'.
    """
    torch.save(states, os.path.join(output_dir, filename))
    if is_best:
        torch.save(states, os.path.join(output_dir, 'checkpoint_best.pth'))


def random_click(mask, point_labels=1, seed=None):
    """
    Generate a random click (x, y) inside the largest label in 'mask'.
    If mask is all zeros, the label is zero.
    """
    max_label = max(set(mask.flatten()))
    if max_label == 0:
        point_labels = 0

    indices = np.argwhere(mask == max_label)
    if seed is not None:
        rand_instance = random.Random(seed)
        rand_num = rand_instance.randint(0, len(indices) - 1)
    else:
        rand_num = random.randint(0, len(indices) - 1)

    output_index_1 = indices[rand_num][0]
    output_index_0 = indices[rand_num][1]
    return point_labels, np.array([output_index_0, output_index_1])


def generate_bbox(mask, variation=0, seed=None):
    """
    Generate a bounding box around the largest label in 'mask'.
    If mask is all zeros, returns an array of nans.
    'variation' can add random scale noise to the bounding box size.
    """
    if seed is not None:
        np.random.seed(seed)
    if len(mask.shape) != 2:
        raise ValueError(f"Mask shape is not 2D, but {mask.shape}")

    max_label = max(set(mask.flatten()))
    if max_label == 0:
        return np.array([np.nan, np.nan, np.nan, np.nan])

    indices = np.argwhere(mask == max_label)
    x0 = np.min(indices[:, 0])
    x1 = np.max(indices[:, 0])
    y0 = np.min(indices[:, 1])
    y1 = np.max(indices[:, 1])
    w = x1 - x0
    h = y1 - y0

    mid_x = (x0 + x1) / 2
    mid_y = (y0 + y1) / 2
    if variation > 0:
        rand_scale = np.random.randn(2) * variation
        w *= (1 + rand_scale[0])
        h *= (1 + rand_scale[1])
        x0 = mid_x - w / 2
        x1 = mid_x + w / 2
        y0 = mid_y - h / 2
        y1 = mid_y + h / 2
    return np.array([y0, x0, y1, x1])


def eval_seg(pred, true_mask_p, threshold):
    """
    Evaluate segmentation performance under various thresholds.
    pred, true_mask_p shape: [B, C, H, W].
    If C == 2, we do a 2-class iou and dice; if C > 2, multi-class; if C == 1, binary seg.
    """
    b, c, h, w = pred.size()
    if c == 2:
        # 2-class segmentation
        iou_d, iou_c, disc_dice, cup_dice = 0, 0, 0, 0
        for th in threshold:
            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            disc_pred = vpred[:, 0, :, :].cpu().numpy().astype('int32')
            cup_pred = vpred[:, 1, :, :].cpu().numpy().astype('int32')

            disc_mask = gt_vmask_p[:, 0, :, :].cpu().numpy().astype('int32')
            cup_mask = gt_vmask_p[:, 1, :, :].cpu().numpy().astype('int32')

            # IOU
            iou_d += iou(disc_pred, disc_mask)
            iou_c += iou(cup_pred, cup_mask)

            # Dice
            disc_dice += dice_coeff(vpred[:, 0, :, :], gt_vmask_p[:, 0, :, :]).item()
            cup_dice += dice_coeff(vpred[:, 1, :, :], gt_vmask_p[:, 1, :, :]).item()

        return (
            iou_d / len(threshold),
            iou_c / len(threshold),
            disc_dice / len(threshold),
            cup_dice / len(threshold)
        )

    elif c > 2:
        # multi-class segmentation: c channels
        ious = [0] * c
        dices = [0] * c
        for th in threshold:
            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            for i_class in range(c):
                class_pred = vpred[:, i_class, :, :].cpu().numpy().astype('int32')
                class_true = gt_vmask_p[:, i_class, :, :].cpu().numpy().astype('int32')

                ious[i_class] += iou(class_pred, class_true)
                dices[i_class] += dice_coeff(
                    vpred[:, i_class, :, :],
                    gt_vmask_p[:, i_class, :, :],
                ).item()

        # returns tuple of length 2*c: (iou_0, iou_1, ..., iou_c-1, dice_0, ..., dice_c-1)
        return tuple(np.array(ious + dices) / len(threshold))

    else:
        # c == 1 => single-channel segmentation
        eiou, edice = 0, 0
        for th in threshold:
            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            disc_pred = vpred[:, 0, :, :].cpu().numpy().astype('int32')
            disc_mask = gt_vmask_p[:, 0, :, :].cpu().numpy().astype('int32')

            eiou += iou(disc_pred, disc_mask)
            edice += dice_coeff(vpred[:, 0, :, :], gt_vmask_p[:, 0, :, :]).item()

        return eiou / len(threshold), edice / len(threshold)


def iou(outputs: np.array, labels: np.array):
    """
    Compute IoU (intersection over union) for arrays of shape [B, H, W].
    """
    SMOOTH = 1e-6
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))
    iou_vals = (intersection + SMOOTH) / (union + SMOOTH)
    return iou_vals.mean()


def dice_coeff(input, target):
    """
    Batchwise Dice coefficient for binary segmentation.
    input, target shapes: [B, H, W].
    """
    if input.is_cuda:
        s = torch.FloatTensor(1).to(input.device).zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, (inp, tgt) in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(inp, tgt)

    return s / (i + 1)


class DiceCoeff(Function):
    """
    Dice coeff for individual examples (2D).
    """
    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 1e-4
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps
        return (2 * self.inter.float() + eps) / self.union.float()

    def backward(self, grad_output):
        input, target = self.saved_variables
        grad_input = grad_target = None
        if self.needs_input_grad[0]:
            # d(Dice)/d(pred)
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        # we don't compute gradient wrt target
        return grad_input, grad_target
