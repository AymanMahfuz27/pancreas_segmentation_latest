#download_setup.sh
#description: Download the SAM2 checkpoint from Hugging Face.
#!/bin/bash
#SBATCH -J medsam2_download
#SBATCH -o medsam2_download.o%j
#SBATCH -e medsam2_download.e%j
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00

# Set working directory
cd /work/09999/aymanmahfuz/ls6

# Load required modules
module purge
module load cuda/12.2

# Initialize conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate medsam2

# Create checkpoints directory
mkdir -p checkpoints

# Download SAM2 checkpoint
wget -P checkpoints https://huggingface.co/jiayuanz3/MedSAM2_pretrain/resolve/main/sam2_hiera_small.pt

# Verify environment and downloads
python3 << END
import torch
import monai
import cv2
import os
import sys

print("\nEnvironment Check:")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")

print("\nCheckpoint Check:")
checkpoint_path = "checkpoints/sam2_hiera_small.pt"
if os.path.exists(checkpoint_path):
    print(f"Checkpoint file exists: {checkpoint_path}")
    print(f"File size: {os.path.getsize(checkpoint_path) / (1024*1024):.2f} MB")
else:
    print("Checkpoint file not found!")
END