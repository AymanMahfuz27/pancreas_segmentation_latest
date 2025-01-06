#!/bin/bash
#SBATCH -J nnunet_train_333
#SBATCH -o nnunet_train_333.o%j
#SBATCH -e nnunet_train_333.e%j
#SBATCH -p gpu-h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aymanmahfuz27@utexas.edu

##################################################################################
# This SLURM script trains 5 folds (0..4) for Task333_T2PancreasMega
# using 3D full-resolution with nnTransUNetTrainerV2.
#
# We allocate 48 hours to hopefully complete all folds. If you suspect more time
# is needed, use a development queue or partition with more time if available.
#
# Note: If you only want fold=0 to reduce time, remove the for-loop over folds.
##################################################################################

module purge
module load tacc-apptainer

export nnUNet_raw_data_base=/scratch/09999/aymanmahfuz/nnUNet_raw_data_base
export nnUNet_preprocessed=/scratch/09999/aymanmahfuz/nnUNet_preprocessed
export RESULTS_FOLDER=/scratch/09999/aymanmahfuz/nnUNet_trained_models

CONTAINER_PATH=/work/09999/aymanmahfuz/ls6/containers/pansegnet_env.sif

TASK_ID=333
TASK_NAME="Task${TASK_ID}_T2PancreasMega"

echo "=== Starting 5-fold training for $TASK_NAME ==="

for fold in 3 4
do
  echo ">>> Training fold $fold..."
  apptainer exec --nv \
    --bind /scratch:/scratch \
    --bind /work:/work \
    $CONTAINER_PATH \
    bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate pansegnet_env && \
             nnUNet_train 3d_fullres nnTransUNetTrainerV2 $TASK_NAME $fold --npz"
  if [ $? -ne 0 ]; then
    echo ">>> ERROR: Fold $fold training failed!"
    exit 1
  fi
done

echo "=== All folds trained successfully! ==="
echo "Check $RESULTS_FOLDER/nnUNet/3d_fullres/$TASK_NAME for model checkpoints."
