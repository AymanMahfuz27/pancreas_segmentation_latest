#!/bin/bash
#SBATCH -J test_train
#SBATCH -o test_train.o%j
#SBATCH -e test_train.e%j
#SBATCH -p gpu-h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aymanmahfuz27@utexas.edu

# Set working directory
SCRATCH_DIR=/scratch/09999/aymanmahfuz
WORK_DIR=/work/09999/aymanmahfuz/ls6
cd $WORK_DIR

# Load required modules
module purge
module load cuda/12.2

# Initialize conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate medsam2

# Set paths
TEST_DATA_DIR=$SCRATCH_DIR/pancreas_test_data
CHECKPOINT_DIR=$WORK_DIR/checkpoints

# Verify test data and checkpoint
if [ ! -d "$TEST_DATA_DIR" ]; then
    echo "Error: Test data directory not found at $TEST_DATA_DIR"
    exit 1
fi

if [ ! -f "$CHECKPOINT_DIR/sam2_hiera_small.pt" ]; then
    echo "Error: Checkpoint file not found at $CHECKPOINT_DIR/sam2_hiera_small.pt"
    exit 1
fi

echo "Starting single-epoch test training..."
python train_3d.py \
    -net sam2 \
    -exp_name pancreas_test_run \
    -sam_ckpt $CHECKPOINT_DIR/sam2_hiera_small.pt \
    -sam_config sam2_hiera_s \
    -image_size 1024 \
    -val_freq 1 \
    -prompt bbox \
    -prompt_freq 2 \
    -dataset pancreas \
    -data_path $TEST_DATA_DIR \
    -batch_size 1 \
    -num_workers 2 \
    -lr 1e-4 \
    -epochs 1 \
    -save_freq 1

# Check training completion
if [ $? -eq 0 ]; then
    echo "Test training completed successfully"
    echo "Please check the logs and validation metrics before proceeding with full training"
else
    echo "Test training failed. Please check the error logs"
    exit 1
fi