#training_submission.sh
#description: Submit a training job for pancreas segmentation.
#!/bin/bash
#SBATCH -J pancreas_train
#SBATCH -o pancreas_train.o%j
#SBATCH -e pancreas_train.e%j
#SBATCH -p gpu-h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aymanmahfuz27@utexas.edu

# Set working directory
SCRATCH_DIR=/scratch/09999/aymanmahfuz
WORK_DIR=/work/09999/aymanmahfuz/ls6

# Load required modules
module purge
module load cuda/12.2

# Initialize conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate medsam2

# Create data directory in scratch (more space)
DATA_DIR=$SCRATCH_DIR/pancreas_data
mkdir -p $DATA_DIR

# Download and verify data
echo "Starting data download and verification..."
python data_processor.py --dest $DATA_DIR

# Verify successful download
if [ ! -d "$DATA_DIR/paired" ]; then
    echo "Error: Data download failed"
    exit 1
fi

NUM_CASES=$(ls $DATA_DIR/paired | wc -l)
if [ $NUM_CASES -lt 614 ]; then
    echo "Error: Not all cases were downloaded. Found $NUM_CASES/614"
    exit 1
fi

echo "Starting training..."
python train_3d.py \
    -net sam2 \
    -exp_name pancreas_medsam2 \
    -sam_ckpt ./checkpoints/sam2_hiera_small.pt \
    -sam_config sam2_hiera_s \
    -image_size 1024 \
    -val_freq 1 \
    -prompt bbox \
    -prompt_freq 2 \
    -dataset pancreas \
    -data_path $DATA_DIR/paired \
    -batch_size 2 \
    -num_workers 4 \
    -lr 1e-4 \
    -epochs 100 \
    -save_freq 5