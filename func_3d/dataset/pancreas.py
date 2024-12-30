import os
import torch
import numpy as np
from torch.utils.data import Dataset
import SimpleITK as sitk
import logging
from pathlib import Path
import random
import math
import torch.nn.functional as F


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PancreasDataset(Dataset):
    """
    A dataset that:
      1) Reads a 3D volume [Z,H,W] + mask [Z,H,W].
      2) Finds the 3D bounding box of the mask + random-crop patch.
      3) Optionally applies data augmentation to the extracted patch.
      4) Resamples the patch to (patch_z, patch_h, patch_w) => then up to (T, 3, 1024, 1024).
      5) Finally arranges dimensions as [T, 3, H, W], matching what SAM2 expects:
         T in dim0, channels=3 in dim1, then (H,W).
    """

    def __init__(
        self,
        args,
        data_path,
        mode="Training",
        prompt="bbox",
        patch_size=(32, 256, 256),  # (Z, Y, X)
        margin=12,
        do_augmentation=True,  # augment only in training
        patches_per_volume = 2,
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

        # Gather all case directories
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
        debug_dir = "/scratch/debug_patches"
        os.makedirs(debug_dir, exist_ok=True)
        volume_idx = idx // self.patches_per_volume
        case_dir = self.cases[volume_idx]
        vol_path = case_dir / "volume.nii.gz"
        msk_path = case_dir / "mask.nii.gz"

        # 1) Load volume & mask
        vol_sitk = sitk.ReadImage(str(vol_path))
        msk_sitk = sitk.ReadImage(str(msk_path))

        vol_arr = sitk.GetArrayFromImage(vol_sitk).astype(np.float32)  # shape=[Z,H,W]
        msk_arr = sitk.GetArrayFromImage(msk_sitk).astype(np.float32)  # shape=[Z,H,W]
        Z, H, W = vol_arr.shape

        # Binarize mask
        msk_arr = (msk_arr > 0.5).astype(np.float32)

        # 2) bounding box for mask + margin
        nz = np.argwhere(msk_arr > 0.5)  # shape=(N,3) => (z,y,x)
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

        # expand
        z_min = max(0, z_min - self.margin)
        z_max = min(Z - 1, z_max + self.margin)
        y_min = max(0, y_min - self.margin)
        y_max = min(H - 1, y_max + self.margin)
        x_min = max(0, x_min - self.margin)
        x_max = min(W - 1, x_max + self.margin)

        # 3) random crop function
        def random_crop(box_min, box_max, desired_size, full_size):
            """
            box_min, box_max: the bounding-box [min, max] for that dimension.
            desired_size: how many voxels we want in this dimension.
            full_size: total dimension size (Z or H or W).
            Returns (start, end) indices.
            """
            box_size = box_max - box_min + 1
            if box_size >= desired_size:
                # We can pick a random start between [box_min, box_max - desired_size]
                max_start = box_max - desired_size + 1
                if max_start < box_min:
                    start = box_min
                else:
                    start = random.randint(box_min, max_start)
            else:
                # If bounding box is smaller than patch
                # allow shifting earlier
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
        # Suppose bounding box is (z_min,z_max, y_min,y_max, x_min,x_max).
        z_center = (z_min + z_max)//2
        y_center = (y_min + y_max)//2
        x_center = (x_min + x_max)//2

        # Then pick patch_z/2 around z_center, etc. A naive approach:
        z_start = max(0, z_center - self.patch_z//2)
        z_end   = min(Z, z_start + self.patch_z)
        x_start = max(0, x_center - self.patch_w//2)
        x_end   = min(W, x_start + self.patch_w)
        y_start = max(0, y_center - self.patch_h//2)
        y_end   = min(H, y_start + self.patch_h)
        

        # random-crop in Z, Y, X
        # z_start, z_end = random_crop(z_min, z_max, self.patch_z, Z)
        # y_start, y_end = random_crop(y_min, y_max, self.patch_h, H)
        # x_start, x_end = random_crop(x_min, x_max, self.patch_w, W)

        # 4) extract patch => shape [pz, ph, pw]
        vol_patch = vol_arr[z_start : z_end + 1, y_start : y_end + 1, x_start : x_end + 1]
        msk_patch = msk_arr[z_start : z_end + 1, y_start : y_end + 1, x_start : x_end + 1]

        # optional augmentation
        if self.do_augmentation:
            vol_patch, msk_patch = self._augment_3d_patch(vol_patch, msk_patch)

        pz, ph, pw = vol_patch.shape  # patch shape

        # We'll create a torch tensor of shape [pz,1,ph,pw] or [pz,3,ph,pw],
        # but we want final shape => [pz,3,ph',pw'] with upsample to 1024 if needed.

        # Convert patch to torch tensor
        vol_tsr = torch.from_numpy(vol_patch)  # [pz,ph,pw]
        # Insert channel dim => [pz,1,ph,pw]
        vol_tsr = vol_tsr.unsqueeze(1)
        # Repeat channel=3 => [pz,3,ph,pw]
        vol_tsr = vol_tsr.repeat(1, 3, 1, 1)

        # Similarly for mask => [pz,1,ph,pw]
        msk_tsr = torch.from_numpy(msk_patch).unsqueeze(1)  # => [pz,1,ph,pw]

        # 5) We want final shape => [pz,3,1024,1024], i.e. upsample in-plane to 1024x1024
        # (assuming pz not upsampled, only the spatial dims)
        # We'll do F.interpolate with (pz remains the same).
        # We'll treat "pz" as batch dimension => we can do .view(...) or permute to fix it.
        # A straightforward approach is to interpret each z-slice as separate items,
        # but let's do a small trick:
        vol_tsr = vol_tsr.permute(1, 0, 2, 3)  
        # now shape => [3, pz, ph, pw], we can pass it to interpolate as if batch=3.
        # But we want the last 2 dims upsampled to 1024 => see mode="bilinear" ignoring the z dimension
        # We'll unify "batch=3, depth=pz" => that's not standard for 2D interpolation. So let's
        # add an extra dimension to treat pz as batch. A simpler approach is "3D interpolation" or "2D interpolation per slice."
        # We'll do 2D interpolation per slice with a loop or reshape.

        # Easiest is a slice-by-slice approach:
        # We'll do a list comprehension to upsample each slice separately => use F.interpolate on shape [1,3,ph,pw].
        upsampled_slices = []
        upsampled_masks  = []
        for z_idx in range(pz):
            # vol_slice => shape [3, ph, pw]
            vol_slice = vol_tsr[:, z_idx, :, :].unsqueeze(0)  # => [1,3,ph,pw]
            # upsample => => [1,3,1024,1024]
            vol_slice_up = F.interpolate(
                vol_slice,
                size=(512, 512),
                mode="bilinear",
                align_corners=False,
            )
            upsampled_slices.append(vol_slice_up.squeeze(0))  # => shape [3,1024,1024]

            # mask => shape [pz,1,ph,pw], but we similarly do slice by slice
            mask_slice = msk_tsr[z_idx].unsqueeze(0)  # => [1,1,ph,pw]
            mask_slice_up = F.interpolate(
                mask_slice,
                size=(512, 512),
                mode="nearest",
            )
            upsampled_masks.append(mask_slice_up.squeeze(0))  # => [1,1024,1024]

        # Now stack them in z dimension => shape [pz,3,1024,1024]
        vol_ups = torch.stack(upsampled_slices, dim=0)  # => [pz,3,1024,1024]
        msk_ups = torch.stack(upsampled_masks, dim=0)   # => [pz,1,1024,1024]

        # 6) Now Sam2 expects shape => [T,3,H,W], i.e. [pz,3,1024,1024] => this is good
        # We'll build the mask_dict => key=z_idx => {0:  [1,1024,1024]}.
        # Then we do the bounding box if prompt="bbox"

        # Final shape references
        final_pz = vol_ups.size(0)
        final_h  = vol_ups.size(2)
        final_w  = vol_ups.size(3)

        mask_dict = {}
        bbox_dict = {}
        # loop over each z in final_pz
        for z_idx in range(final_pz):
            # shape => [1, 1024,1024]
            slice_mask = msk_ups[z_idx]  # => [1,1024,1024]
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
                    yma, xma = final_h - 1, final_w - 1
                bb = torch.tensor([xmi, ymi, xma, yma], dtype=torch.float32).unsqueeze(0)
                bbox_dict[z_idx] = {0: bb}

        case_name = str(case_dir.name)
        # if random.random() < 0.01:
        #     # Save the patch volume as .npy or .nii
        #     np.save(os.path.join(debug_dir, f"{case_name}_volPatch_{idx}.npy"), vol_patch)
        #     np.save(os.path.join(debug_dir, f"{case_name}_mskPatch_{idx}.npy"), msk_patch)
        #     vol_min, vol_max = vol_patch.min(), vol_patch.max()
        #     msk_sum = msk_patch.sum()
        #     print(f"[DEBUG] case={case_name}, idx={idx}, shape={vol_patch.shape}, vol_min={vol_min}, vol_max={vol_max}, mask_sum={msk_sum}")


        return {
            "image": vol_ups,  # [pz, 3, 1024,1024]
            "label": mask_dict,
            "bbox": bbox_dict if self.prompt == "bbox" else None,
            "image_meta_dict": {"filename_or_obj": [case_name]},
        }

    def _augment_3d_patch(self, vol_patch, msk_patch):
        """
        Simple 3D augmentations: random flips, intensity scale, noise, rotate in-plane.
        """
        import scipy.ndimage

        pz, ph, pw = vol_patch.shape

        # random flips
        if random.random()<0.5:
            vol_patch = vol_patch[::-1, :, :]
            msk_patch = msk_patch[::-1, :, :]
        if random.random()<0.5:
            vol_patch = vol_patch[:, ::-1, :]
            msk_patch = msk_patch[:, ::-1, :]
        if random.random()<0.5:
            vol_patch = vol_patch[:, :, ::-1]
            msk_patch = msk_patch[:, :, ::-1]

        # fix negative strides
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

        # random rotation about z-axis (2D rotation slice by slice)
        angle_deg = random.uniform(-15, 15)
        angle_rad = angle_deg * math.pi/180.0
        if random.random() < 0.5:
            vol_patch, msk_patch = self._rotate_in_plane(vol_patch, msk_patch, angle_rad)
            vol_patch = np.ascontiguousarray(vol_patch)
            msk_patch = np.ascontiguousarray(msk_patch)

        return vol_patch, msk_patch

    def _rotate_in_plane(self, vol_patch, msk_patch, angle_rad):
        import scipy.ndimage
        pz, ph, pw = vol_patch.shape
        angle_deg = angle_rad*180.0/math.pi

        for z in range(pz):
            vol_slice = vol_patch[z]
            vol_slice_rot = scipy.ndimage.rotate(
                vol_slice,
                angle=angle_deg,
                axes=(1,0),
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
                axes=(1,0),
                reshape=False,
                order=0,
                cval=0.0,
                prefilter=False
            )
            msk_patch[z] = (msk_slice_rot>0.5).astype(msk_patch.dtype)
        return vol_patch, msk_patch
