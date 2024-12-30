#!/bin/bash
#SBATCH -J full_train
#SBATCH -o full_train.o%j
#SBATCH -e full_train.e%j
#SBATCH -p gpu-h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aymanmahfuz27@utexas.edu

PROJECT_DIR=/home1/09999/aymanmahfuz/pancreas_project_latest/pancreas_segmentation_latest
WORK_DIR=/work/09999/aymanmahfuz/ls6
SCRATCH_DIR=/scratch/09999/aymanmahfuz
CONTAINER_PATH=$WORK_DIR/containers/medsam2.sif

mkdir -p $SCRATCH_DIR/model_checkpoints

module purge
module load tacc-apptainer

cd $PROJECT_DIR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


echo "Starting full training..."

apptainer exec --nv \
  --bind $PROJECT_DIR:/project \
  --bind $WORK_DIR:/work \
  --bind $SCRATCH_DIR:/scratch \
  $CONTAINER_PATH \
  bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate medsam2 && \
           python /project/train_3d.py \
             -net sam2 \
             -exp_name pancreas_full_training \
             -sam_ckpt /work/checkpoints/sam2_hiera_small.pt \
             -sam_config sam2_hiera_s \
             -image_size 1024 \
             -val_freq 5 \
             -prompt bbox \
             -prompt_freq 2 \
             -dataset pancreas \
             -data_path /scratch/pancreas_test_data_preproc \
             -b 1 \
             -lr 1e-4 \
             -multimask_output 0 \
             -memory_bank_size 32 \
             -distributed 0"

if [ $? -eq 0 ]; then
    echo "Training completed successfully"
    # Tar up results, etc.
else
    echo "Training failed. Check logs."
    exit 1
fi
