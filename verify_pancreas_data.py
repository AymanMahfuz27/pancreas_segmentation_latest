import os
import torch
import numpy as np
from pathlib import Path
import pydicom
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_dicom_data(data_dir, output_dir):
    """Verify DICOM data integrity and characteristics."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Get all case directories
    cases = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    logger.info(f"Found {len(cases)} cases")
    
    # Statistics containers
    stats = {
        'mri_shapes': [],
        'mask_shapes': [],
        'mri_ranges': [],
        'mask_ranges': [],
        'mask_unique_values': []
    }
    
    # Analyze first 5 cases in detail
    for case_dir in tqdm(cases[:5], desc="Analyzing cases"):
        mri_path = case_dir / 'mri.dcm'
        mask_path = case_dir / 'mask.dcm'
        
        try:
            # Load DICOM files
            mri_dcm = pydicom.dcmread(str(mri_path))
            mask_dcm = pydicom.dcmread(str(mask_path))
            
            mri = mri_dcm.pixel_array
            mask = mask_dcm.pixel_array
            
            # Record statistics
            stats['mri_shapes'].append(mri.shape)
            stats['mask_shapes'].append(mask.shape)
            stats['mri_ranges'].append((float(mri.min()), float(mri.max())))
            stats['mask_ranges'].append((float(mask.min()), float(mask.max())))
            stats['mask_unique_values'].append(np.unique(mask).tolist())
            
            # Save visualizations for middle slice
            if len(mri.shape) == 3:
                middle_slice = mri.shape[0] // 2
                mri_slice = mri[middle_slice]
                mask_slice = mask[middle_slice]
            else:
                mri_slice = mri
                mask_slice = mask
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            ax1.imshow(mri_slice, cmap='gray')
            ax1.set_title(f'MRI (range: [{mri_slice.min():.1f}, {mri_slice.max():.1f}])')
            ax2.imshow(mask_slice, cmap='gray')
            ax2.set_title(f'Mask (unique values: {np.unique(mask_slice)})')
            plt.suptitle(f'Case: {case_dir.name}')
            plt.savefig(output_dir / f'{case_dir.name}_visualization.png')
            plt.close()
            
            logger.info(f"\nCase {case_dir.name}:")
            logger.info(f"MRI shape: {mri.shape}")
            logger.info(f"Mask shape: {mask.shape}")
            logger.info(f"MRI range: [{mri.min()}, {mri.max()}]")
            logger.info(f"Mask unique values: {np.unique(mask)}")
            logger.info(f"Mask coverage: {(mask > 0).sum() / mask.size * 100:.2f}%")
            
        except Exception as e:
            logger.error(f"Error processing case {case_dir}: {e}")
    
    # Print summary statistics
    logger.info("\nSummary Statistics:")
    logger.info(f"Unique MRI shapes: {set(stats['mri_shapes'])}")
    logger.info(f"Unique mask shapes: {set(stats['mask_shapes'])}")
    logger.info(f"MRI value ranges: min={min(r[0] for r in stats['mri_ranges'])}, max={max(r[1] for r in stats['mri_ranges'])}")
    logger.info(f"Mask value ranges: min={min(r[0] for r in stats['mask_ranges'])}, max={max(r[1] for r in stats['mask_ranges'])}")
    logger.info(f"All unique mask values: {set(sum(stats['mask_unique_values'], []))}")

    # Now load the PancreasDataset to check processed output
    from func_3d.dataset.pancreas import PancreasDataset
    class Args:
        image_size = 1024
        video_length = 2

    args = Args()
    dataset = PancreasDataset(args, str(data_dir), mode='Training')
    logger.info(f"\nPancreasDataset length: {len(dataset)}")
    if len(dataset) > 0:
        sample = dataset[0]
        image = sample['image']  # [T,C,H,W]
        mask_dict = sample['label']  # {t: {0: mask}}
        mask_tensor = mask_dict[0][0]  # first frame, first mask
        logger.info(f"Sample image shape: {image.shape}")
        logger.info(f"Sample mask shape: {mask_tensor.shape}")
        logger.info(f"Sample mask unique values: {torch.unique(mask_tensor)}")

        # Print first frame min/max to confirm normalization
        first_frame = image[0,0]  # first time, first channel
        logger.info(f"First frame MRI range after normalization: [{float(first_frame.min())}, {float(first_frame.max())}]")

        # Confirm mask binary nature
        logger.info(f"Mask is binary? {set(torch.unique(mask_tensor).tolist()) <= {0.0,1.0}}")

if __name__ == "__main__":
    verify_dicom_data(
        data_dir="/scratch/pancreas_test_data",
        output_dir="/scratch/data_verification"
    )
