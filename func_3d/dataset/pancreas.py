import os
import torch
import numpy as np
from torch.utils.data import Dataset
import pydicom
from pathlib import Path
import logging
from PIL import Image
import torchvision.transforms.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PancreasDataset(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='Training', prompt='bbox'):
        self.args = args
        self.data_path = Path(data_path)
        self.transform = transform
        self.transform_msk = transform_msk
        self.mode = mode
        self.prompt = prompt
        self.video_length = args.video_length
        
        # Get all case directories
        all_cases = sorted([d for d in self.data_path.iterdir() if d.is_dir()])
        
        # Split cases into training and testing sets (80-20 split)
        n_cases = len(all_cases)
        n_train = int(0.8 * n_cases)
        
        if mode == 'Training':
            self.cases = all_cases[:n_train]
        else:
            self.cases = all_cases[n_train:]
            
        logger.info(f"Found {len(self.cases)} cases for {mode}")

    def normalize_and_resize_slice(self, slice_array):
        """Normalize, convert to PIL, and resize a single slice."""
        # slice_array: [H,W], float32
        if slice_array.max() > 0:
            slice_array = slice_array / slice_array.max()
        
        slice_pil = Image.fromarray((slice_array * 255).astype(np.uint8))
        slice_pil = F.resize(slice_pil, (self.args.image_size, self.args.image_size))
        slice_resized = np.array(slice_pil, dtype=np.float32) / 255.0
        return slice_resized

    def select_slices(self, total_slices, required_slices):
        """Select indices of slices to form the video_length.

        If total_slices == required_slices, use all.
        If total_slices > required_slices, sample evenly.
        If total_slices < required_slices, repeat some slices.
        """
        if total_slices == required_slices:
            return list(range(total_slices))
        elif total_slices > required_slices:
            # Evenly sample required_slices from total_slices
            return np.round(np.linspace(0, total_slices-1, required_slices)).astype(int).tolist()
        else:
            # total_slices < required_slices, repeat slices
            # Just cycle through slices until we have enough
            repeats = required_slices // total_slices
            remainder = required_slices % total_slices
            return list(range(total_slices)) * repeats + list(range(remainder))

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case_dir = self.cases[idx]
        try:
            # Load MRI and mask DICOM files
            mri_path = case_dir / 'mri.dcm'
            mask_path = case_dir / 'mask.dcm'
            
            mri_dcm = pydicom.dcmread(str(mri_path))
            mask_dcm = pydicom.dcmread(str(mask_path))
            
            # MRI and Mask are likely 3D volumes: (T,H,W)
            mri_vol = mri_dcm.pixel_array.astype(np.float32)
            mask_vol = mask_dcm.pixel_array.astype(np.float32)
            
            # Check dimensions
            # Typically shape is (T,H,W). If there's extra dimensions, squeeze them out.
            mri_vol = np.squeeze(mri_vol)
            mask_vol = np.squeeze(mask_vol)
            
            if len(mri_vol.shape) != 3:
                raise ValueError(f"MRI volume expected 3D but got {mri_vol.shape}")
            if len(mask_vol.shape) != 3:
                raise ValueError(f"Mask volume expected 3D but got {mask_vol.shape}")

            T_vol, H, W = mri_vol.shape
            tm, hm, wm = mask_vol.shape
            if (tm, hm, wm) != (T_vol, H, W):
                raise ValueError("MRI and mask volume shapes do not match")

            # Select slices
            slice_indices = self.select_slices(T_vol, self.video_length)

            # Process MRI slices
            mri_slices = []
            for i in slice_indices:
                slice_2d = mri_vol[i]  # [H,W]
                slice_resized = self.normalize_and_resize_slice(slice_2d)
                # slice_resized: [H,W]
                mri_slices.append(slice_resized)

            # mri_slices: [video_length, H, W]
            mri_slices = np.stack(mri_slices, axis=0)  # [T,H,W]

            # Convert to 3-channel
            mri_slices = np.stack([mri_slices]*3, axis=1)  # [T,C,H,W] with C=3

            # Process Mask slices
            mask_slices = []
            for i in slice_indices:
                slice_2d = mask_vol[i]
                # Normalize mask? Typically masks are binary or label maps
                # Just resize directly
                mask_pil = Image.fromarray(slice_2d.astype(np.uint8))
                mask_pil = F.resize(mask_pil, (self.args.image_size, self.args.image_size), interpolation=Image.NEAREST)
                mask_resized = np.array(mask_pil, dtype=np.float32)
                # [H,W]
                mask_slices.append(mask_resized)

            mask_slices = np.stack(mask_slices, axis=0)  # [T,H,W]

            # Convert to torch
            mri = torch.from_numpy(mri_slices)  # [T,3,H,W]
            mask = torch.from_numpy(mask_slices).unsqueeze(1)  # [T,1,H,W]

            # Generate bounding box prompt from the first frame or all frames?
            # Typically we can generate from each frame. The code expects a dict of frames.
            # Let's use the first frame with the bounding box (or each frame).
            mask_dict = {}
            bbox_dict = {}
            for t in range(self.video_length):
                frame_mask = mask[t]  # [1,H,W]
                mask_dict[t] = {0: frame_mask}

                if self.prompt == 'bbox':
                    mask_2d = frame_mask.squeeze(0)
                    nonzero_indices = torch.nonzero(mask_2d)
                    if nonzero_indices.numel() > 0:
                        y_min = nonzero_indices[:, 0].min()
                        y_max = nonzero_indices[:, 0].max()
                        x_min = nonzero_indices[:, 1].min()
                        x_max = nonzero_indices[:, 1].max()
                        bbox = torch.tensor([x_min, y_min, x_max, y_max]).float().unsqueeze(0) # [1,4]
                    else:
                        bbox = torch.tensor([[0, 0, mask_2d.shape[1]-1, mask_2d.shape[0]-1]], dtype=torch.float32)
                    bbox_dict[t] = {0: bbox}

            if self.prompt == 'bbox':
                bbox_final = bbox_dict
            else:
                bbox_final = None

            return {
                'image': mri,  # [T,C,H,W]
                'label': mask_dict,
                'bbox': bbox_final,
                'image_meta_dict': {'filename_or_obj': [str(case_dir.name)]}
            }
            
        except Exception as e:
            logger.error(f"Error loading case {case_dir}: {str(e)}")
            raise
