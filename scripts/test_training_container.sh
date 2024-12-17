#!/bin/bash
#SBATCH -J train_medsam2
#SBATCH -o train_medsam2.o%j
#SBATCH -e train_medsam2.e%j
#SBATCH -p gpu-h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00

# Set directories
WORK_DIR=/work/09999/aymanmahfuz/ls6
PROJECT_DIR=/home1/09999/aymanmahfuz/pancreas_project_latest/pancreas_segmentation_latest
SCRATCH_DIR=/scratch/09999/aymanmahfuz
CONTAINER_PATH=$WORK_DIR/containers/medsam2.sif

# Load required modules
module load tacc-singularity
module load cuda/12.2

# Create bind path string
BIND_PATHS="$WORK_DIR,$SCRATCH_DIR,$PROJECT_DIR"

# Run training inside container
singularity exec --nv \
    --bind $BIND_PATHS \
    $CONTAINER_PATH \
    python $PROJECT_DIR/train_3d.py \
    -net sam2 \
    -exp_name pancreas_test_run \
    -sam_ckpt $WORK_DIR/checkpoints/sam2_hiera_small.pt \
    -sam_config sam2_hiera_t \
    -image_size 1024 \
    -val_freq 1 \
    -prompt bbox \
    -prompt_freq 2 \
    -dataset pancreas \
    -data_path $SCRATCH_DIR/pancreas_test_data \
    -b 1 \
    -lr 1e-4 \
    -multimask_output 0 \
    -memory_bank_size 32 \
    -distributed 0