#!/bin/bash
#SBATCH -J TestGPU           # Job name
#SBATCH -o TestGPU.o%j       # Name of stdout output file (%j expands to jobId)
#SBATCH -e TestGPU.e%j       # Name of stderr error file (%j expands to jobId)
#SBATCH -p gpu-a100          # Queue name
#SBATCH -N 1                 # Total number of nodes
#SBATCH -n 1                 # Total number of mpi tasks
#SBATCH -t 00:10:00         # Run time (hh:mm:ss)

# Load required modules
module purge
module load cuda/12.2
module load python3/3.9.7

# Initialize conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate medsam2

# Echo job information
echo "Job started at $(date)"
echo "Running on host $(hostname)"
pwd

# Check GPU availability
nvidia-smi

# Test PyTorch GPU access
python3 -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Number of GPUs:', torch.cuda.device_count()); print('GPU Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"

# Print environment information
echo "Python version:"
python3 --version
echo "Conda environment:"
conda info
echo "Job finished at $(date)"