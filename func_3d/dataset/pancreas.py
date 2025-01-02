import os
import torch
import numpy as np
import random
import math
import logging
import SimpleITK as sitk
from pathlib import Path
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PancreasDataset(Dataset):
    """
    A dataset that:
      1) Reads a 3D volume [Z,H,W] + binary mask [Z,H,W].
      2) Finds the 3D bounding box of the mask + margin.
      3) Randomly crops out a patch of shape (patch_z, 512, 512) (no up/down-sampling).
      4) Optionally applies 3D augmentation (flips, noise, etc.).
      5) Replicates channels => [pz, 3, 512, 512].
      6) Returns per-slice dict for SAM2.

    NOTE: This ensures that the spatial in-plane size is exactly 512×512, matching
    your `image_size=512` in SAM2Base. That way, the final feature map from the
    backbone is 32×32 (since 512/16 = 32).
    """

    def __init__(
        self,
        args,
        data_path,
        mode="Training",
        prompt="bbox",
        # IMPORTANT: patch_size now has 512 for Y, X so that we don't do any 256→512 upscaling
        patch_size=(32, 512, 512),  # (Z, Y, X)
        margin=12,
        do_augmentation=True, 
        patches_per_volume=4
    ):
        super().__init__()
        self.args = args
        self.data_path = Path(data_path)
        self.mode = mode
        self.prompt = prompt

        self.patch_z, self.patch_h, self.patch_w = patch_size
        self.margin = margin
        self.do_augmentation = do_augmentation and (mode == "Training")
        self.patches_per_volume = patches_per_volume

        # Gather case directories
        all_cases = sorted([d for d in self.data_path.iterdir() if d.is_dir()])
        n_cases = len(all_cases)
        n_train = int(0.8 * n_cases)
        if mode == "Training":
            self.cases = all_cases[:n_train]
        else:
            self.cases = all_cases[n_train:]

        logger.info(f"[{mode}] Found {len(self.cases)} cases at {data_path}")

    def __len__(self):
        return len(self.cases) * self.patches_per_volume

    def __getitem__(self, idx):
        volume_idx = idx // self.patches_per_volume
        case_dir = self.cases[volume_idx]
        vol_path = case_dir / "volume.nii.gz"
        msk_path = case_dir / "mask.nii.gz"

        # 1) Load volume & mask => shape [Z,H,W]
        vol_sitk = sitk.ReadImage(str(vol_path))
        msk_sitk = sitk.ReadImage(str(msk_path))
        vol_arr = sitk.GetArrayFromImage(vol_sitk).astype(np.float32)
        msk_arr = sitk.GetArrayFromImage(msk_sitk).astype(np.float32)

        # Binarize mask
        msk_arr = (msk_arr > 0.5).astype(np.float32)
        Z, H, W = vol_arr.shape

        # 2) bounding box + margin
        nz = np.argwhere(msk_arr > 0.5)
        if nz.size == 0:
            z_min, z_max = 0, Z - 1
            y_min, y_max = 0, H - 1
            x_min, x_max = 0, W - 1
        else:
            z_vals = nz[:, 0]
            y_vals = nz[:, 1]
            x_vals = nz[:, 2]
            z_min, z_max = z_vals.min(), z_vals.max()
            y_min, y_max = y_vals.min(), y_vals.max()
            x_min, x_max = x_vals.min(), x_vals.max()

        # expand bounding box by margin
        z_min = max(0, z_min - self.margin)
        z_max = min(Z - 1, z_max + self.margin)
        y_min = max(0, y_min - self.margin)
        y_max = min(H - 1, y_max + self.margin)
        x_min = max(0, x_min - self.margin)
        x_max = min(W - 1, x_max + self.margin)

        # 3) random-crop in 1D
        def random_crop_1D(box_min, box_max, desired_size, full_size):
            box_size = box_max - box_min + 1
            if box_size >= desired_size:
                max_start = box_max - desired_size + 1
                if max_start < box_min:
                    start = box_min
                else:
                    start = random.randint(box_min, max_start)
            else:
                extra = desired_size - box_size
                min_start = max(0, box_min - extra)
                max_start = box_min
                if min_start > max_start:
                    start = box_min
                else:
                    start = random.randint(min_start, max_start)
            end = start + desired_size - 1
            if end >= full_size:
                end = full_size - 1
                start = end - desired_size + 1
            return start, end

        # Crop in z, y, x
        z_start, z_end = random_crop_1D(z_min, z_max, self.patch_z, Z)
        y_start, y_end = random_crop_1D(y_min, y_max, self.patch_h, H)
        x_start, x_end = random_crop_1D(x_min, x_max, self.patch_w, W)

        # 4) extract patch => shape [pz, ph, pw]
        vol_patch = vol_arr[z_start : z_end + 1, y_start : y_end + 1, x_start : x_end + 1]
        msk_patch = msk_arr[z_start : z_end + 1, y_start : y_end + 1, x_start : x_end + 1]
        pz, ph, pw = vol_patch.shape

        # optional 3D augmentation
        if self.do_augmentation:
            vol_patch, msk_patch = self._augment_3d_patch(vol_patch, msk_patch)

        # 5) replicate channels => shape => [pz, 3, ph, pw]
        vol_tsr = torch.from_numpy(vol_patch)  # => [pz, ph, pw]
        vol_tsr = vol_tsr.unsqueeze(1).repeat(1, 3, 1, 1)  # => [pz, 3, ph, pw]

        msk_tsr = torch.from_numpy(msk_patch).unsqueeze(1)  # => [pz, 1, ph, pw]

        # Build mask_dict (and optional bounding boxes)
        mask_dict = {}
        bbox_dict = {}
        for z_idx in range(pz):
            slice_mask = msk_tsr[z_idx]  # => shape [1,ph,pw]
            mask_dict[z_idx] = {0: slice_mask}
            if self.prompt == "bbox":
                nz_idx = (slice_mask[0] > 0).nonzero()
                if nz_idx.numel() > 0:
                    ymi = nz_idx[:, 0].min()
                    yma = nz_idx[:, 0].max()
                    xmi = nz_idx[:, 1].min()
                    xma = nz_idx[:, 1].max()
                else:
                    ymi, xmi = 0, 0
                    yma, xma = ph - 1, pw - 1
                bb = torch.tensor([xmi, ymi, xma, yma], dtype=torch.float32).unsqueeze(0)
                bbox_dict[z_idx] = {0: bb}

        case_name = str(case_dir.name)
        return {
            "image": vol_tsr,   # shape => [pz, 3, 512, 512]
            "label": mask_dict,
            "bbox": bbox_dict if self.prompt == "bbox" else None,
            "image_meta_dict": {"filename_or_obj": [case_name]},
        }

    def _augment_3d_patch(self, vol_patch, msk_patch):
        """
        Random flips, intensity scaling, gaussian noise, rotation in-plane, etc.
        vol_patch, msk_patch => shape [pz, ph, pw].
        """
        import numpy as np
        import random, math
        import scipy.ndimage

        pz, ph, pw = vol_patch.shape

        # random flips
        if random.random() < 0.5:
            vol_patch = vol_patch[::-1, :, :]
            msk_patch = msk_patch[::-1, :, :]
        if random.random() < 0.5:
            vol_patch = vol_patch[:, ::-1, :]
            msk_patch = msk_patch[:, ::-1, :]
        if random.random() < 0.5:
            vol_patch = vol_patch[:, :, ::-1]
            msk_patch = msk_patch[:, :, ::-1]

        # ensure contiguity
        vol_patch = np.ascontiguousarray(vol_patch)
        msk_patch = np.ascontiguousarray(msk_patch)

        # intensity scaling
        scale_factor = random.uniform(0.8, 1.2)
        vol_patch *= scale_factor

        # random gaussian noise
        if random.random() < 0.2:
            stdev = vol_patch.std()
            if stdev < 1e-6:
                stdev = 0.01
            noise = np.random.randn(*vol_patch.shape).astype(vol_patch.dtype) * (0.05 * stdev)
            vol_patch = vol_patch + noise

        # random rotation in-plane
        angle_deg = random.uniform(-15, 15)
        if random.random() < 0.5:
            angle_rad = angle_deg * math.pi / 180.0
            for z in range(pz):
                vol_slice = vol_patch[z]
                vol_slice_rot = scipy.ndimage.rotate(
                    vol_slice,
                    angle=angle_deg,
                    axes=(1, 0),
                    reshape=False,
                    order=1,
                    cval=0.0,
                    prefilter=False
                )
                vol_patch[z] = vol_slice_rot

                msk_slice = msk_patch[z]
                msk_slice_rot = scipy.ndimage.rotate(
                    msk_slice,
                    angle=angle_deg,
                    axes=(1, 0),
                    reshape=False,
                    order=0,
                    cval=0.0,
                    prefilter=False
                )
                msk_patch[z] = (msk_slice_rot > 0.5).astype(msk_patch.dtype)

        return vol_patch, msk_patch
