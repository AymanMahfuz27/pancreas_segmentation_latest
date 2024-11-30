#!/bin/bash
#SBATCH -J setup_checkpoint
#SBATCH -o setup_checkpoint.o%j
#SBATCH -e setup_checkpoint.e%j
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00

cd /work/09999/aymanmahfuz/ls6

# Load required modules
module purge
module load cuda/12.2

# Initialize conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate medsam2

# Create checkpoints directory if it doesn't exist
mkdir -p checkpoints

# Copy and rename the checkpoint file
cp MedSAM2_pretrain/MedSAM2_pretrain.pth checkpoints/sam2_hiera_small.pt

# Verify the setup
python3 << END
import os
import torch

checkpoint_path = "checkpoints/sam2_hiera_small.pt"
if os.path.exists(checkpoint_path):
    size_mb = os.path.getsize(checkpoint_path) / (1024*1024)
    print(f"\nCheckpoint verification:")
    print(f"Location: {checkpoint_path}")
    print(f"File size: {size_mb:.2f} MB")
    
    try:
        # Attempt to load the checkpoint to verify its integrity
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print("Successfully loaded checkpoint into PyTorch")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
else:
    print("\nError: Checkpoint file not found!")
END