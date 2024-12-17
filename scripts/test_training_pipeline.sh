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

# Set directories
PROJECT_DIR=/home1/09999/aymanmahfuz/pancreas_project_latest/pancreas_segmentation_latest
WORK_DIR=/work/09999/aymanmahfuz/ls6
SCRATCH_DIR=/scratch/09999/aymanmahfuz

# Change to project directory
cd $PROJECT_DIR

# Load required modules
module purge
module load cuda/12.2

# Initialize conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate medsam2

# Set Python path to include project directories
export PYTHONPATH=$PROJECT_DIR:$PROJECT_DIR/sam2_train:$PYTHONPATH

# Configure data paths
TEST_DATA_DIR=$SCRATCH_DIR/pancreas_test_data
CHECKPOINT_DIR=$WORK_DIR/checkpoints

# Print configuration for verification
echo "Configuration:"
echo "Project Directory: $PROJECT_DIR"
echo "Work Directory: $WORK_DIR"
echo "Test Data Directory: $TEST_DATA_DIR"
echo "Checkpoint Path: $CHECKPOINT_DIR/sam2_hiera_small.pt"
echo "Python Path: $PYTHONPATH"
echo "Config File: sam2_hiera_s"


# Create python script to modify train_3d.py temporarily
cat << 'EOF' > modify_train.py
import fileinput
import sys

filename = 'train_3d.py'
search_text = 'loss = function.train_sam(args, net, optimizer, nice_train_loader, epoch)'
replace_text = 'loss = function.train_sam(args, net, optimizer, None, nice_train_loader, epoch)'

with fileinput.FileInput(filename, inplace=True, backup='.bak') as file:
    for line in file:
        print(line.replace(search_text, replace_text), end='')
EOF

# Apply the modification
python modify_train.py


# Begin training
echo "Starting single-epoch test training..."
python train_3d.py \
    -net sam2 \
    -exp_name pancreas_test_run \
    -sam_ckpt $CHECKPOINT_DIR/sam2_hiera_small.pt \
    -sam_config sam2_hiera_t \
    -image_size 1024 \
    -val_freq 1 \
    -prompt bbox \
    -prompt_freq 2 \
    -dataset pancreas \
    -data_path $TEST_DATA_DIR \
    -b 1 \
    -lr 1e-4 \
    -multimask_output 0 \
    -memory_bank_size 32 \
    -distributed 0

# Restore the original train_3d.py
mv train_3d.py.bak train_3d.py


# Check completion status
if [ $? -eq 0 ]; then
    echo "Test training completed successfully"
else
    echo "Test training failed. Please check the error logs"
    exit 1
fi