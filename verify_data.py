import torch
import numpy as np
from pathlib import Path
import pydicom
import matplotlib.pyplot as plt
from func_3d.dataset.pancreas import PancreasDataset
import argparse

def analyze_dataset(data_path, output_dir):
    class Args:
        def __init__(self):
            self.image_size = 1024
            self.video_length = 2

    args = Args()
    dataset = PancreasDataset(args, data_path, mode='Training')
    
    print(f"\nAnalyzing {len(dataset)} cases...")
    
    # Statistics containers
    mri_stats = {'min': [], 'max': [], 'mean': [], 'std': []}
    mask_stats = {'unique_values': [], 'coverage': []}
    
    # Sample and analyze first 5 cases
    for i in range(min(5, len(dataset))):
        data = dataset[i]
        mri = data['image']  # [T,C,H,W]
        mask_dict = data['label']
        
        # Get first frame mask
        mask = mask_dict[0][0]  # Get first frame, first object mask
        
        # MRI statistics
        mri_frame = mri[0, 0]  # First frame, first channel
        mri_stats['min'].append(float(mri_frame.min()))
        mri_stats['max'].append(float(mri_frame.max()))
        mri_stats['mean'].append(float(mri_frame.mean()))
        mri_stats['std'].append(float(mri_frame.std()))
        
        # Mask statistics
        unique_vals = torch.unique(mask).tolist()
        mask_coverage = (mask > 0).float().mean().item()
        mask_stats['unique_values'].append(unique_vals)
        mask_stats['coverage'].append(mask_coverage)
        
        # Save visualization
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(mri_frame.numpy(), cmap='gray')
        axes[0].set_title(f'MRI Frame (min={mri_frame.min():.2f}, max={mri_frame.max():.2f})')
        axes[1].imshow(mask[0].numpy(), cmap='gray')
        axes[1].set_title(f'Mask (coverage={mask_coverage*100:.2f}%)')
        plt.savefig(f'{output_dir}/case_{i}_visualization.png')
        plt.close()
        
        print(f"\nCase {i}:")
        print(f"MRI shape: {mri.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"MRI range: [{mri_frame.min():.2f}, {mri_frame.max():.2f}]")
        print(f"Unique mask values: {unique_vals}")
        print(f"Mask coverage: {mask_coverage*100:.2f}%")
        
        if i == 0:
            # Save raw arrays for first case
            np.save(f'{output_dir}/first_case_mri.npy', mri.numpy())
            np.save(f'{output_dir}/first_case_mask.npy', mask.numpy())
    
    print("\nOverall Statistics:")
    print("MRI:")
    for key in mri_stats:
        print(f"{key}: mean={np.mean(mri_stats[key]):.3f}, std={np.std(mri_stats[key]):.3f}")
    print("\nMask:")
    print(f"Average coverage: {np.mean(mask_stats['coverage'])*100:.2f}%")
    print(f"All unique values found: {set(sum(mask_stats['unique_values'], []))}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    analyze_dataset(args.data_path, args.output_dir)
