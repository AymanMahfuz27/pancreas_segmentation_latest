#!/bin/bash
#SBATCH -J panseg_t2_training
#SBATCH -o panseg_t2_training.o%j
#SBATCH -e panseg_t2_training.e%j
#SBATCH -p gpu-h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aymanmahfuz27@utexas.edu

##################################################################################
# This SLURM script trains a 3D U-Net (via nnU-Net) on T2 pancreas data
# combined from your local dataset + PaNSegNet T2. We use 5-fold cross-validation
# for robust performance and to auto-tune postprocessing. The final model is
# typically found via nnUNet_find_best_configuration after training.
#
# Steps:
#   1. Export environment variables for nnU-Net
#   2. Plan & preprocess with nnUNet_plan_and_preprocess (Task ID=333)
#   3. Train 3D fullres with the nnTransUNetTrainerV2 trainer (from PaNSegNet)
#   4. Train folds 0..4 in a loop
#   5. Identify best fold or do an ensemble
#
# Future Researchers:
# - If you want to tweak training or reduce time, skip some folds.
# - If you want to prevent overfitting, rely on the 5-fold CV and nnU-Net's
#   data augmentation. You can also watch the "progress.png" to see if it's converging.
##################################################################################

# 1) Load Apptainer (Singularity)
module purge
module load tacc-apptainer

# 2) Export nnU-Net environment variables (where data & results go)
export nnUNet_raw_data_base=/scratch/09999/aymanmahfuz/nnUNet_raw_data_base
export nnUNet_preprocessed=/scratch/09999/aymanmahfuz/nnUNet_preprocessed
export RESULTS_FOLDER=/scratch/09999/aymanmahfuz/nnUNet_trained_models

# 3) Path to your container
CONTAINER_PATH=/work/09999/aymanmahfuz/ls6/containers/pansegnet_env.sif

# 4) Chosen Task ID & name (arbitrary ID: 333). Must match folder name in nnUNet_raw_data
TASK_ID=333
TASK_NAME="Task${TASK_ID}_T2PancreasMega"  # Must match your folder name inside nnUNet_raw_data

# 5) Plan & preprocess
echo "=== Step 1: nnU-Net Plan & Preprocess for $TASK_NAME ==="
apptainer exec --nv \
  --bind /scratch:/scratch \
  --bind /work:/work \
  $CONTAINER_PATH \
  bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate pansegnet_env && \
           nnUNet_plan_and_preprocess -t $TASK_ID"

if [ $? -ne 0 ]; then
  echo ">>> ERROR: plan_and_preprocess failed."
  exit 1
fi

# 6) Train Folds (0..4) with 3D fullres & custom trainer "nnTransUNetTrainerV2"
#    This trains 5 separate models (cross-validation).
#    If you only want 1 final model (less time), you can skip folds 1..4.
echo "=== Step 2: Training folds with 3D Fullres + nnTransUNetTrainerV2 ==="

for fold in 0 1 2 3 4
do
  echo ">>> Training fold $fold..."
  apptainer exec --nv \
    --bind /scratch:/scratch \
    --bind /work:/work \
    $CONTAINER_PATH \
    bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate pansegnet_env && \
             nnUNet_train 3d_fullres nnTransUNetTrainerV2 $TASK_NAME $fold --npz"
  if [ $? -ne 0 ]; then
    echo ">>> ERROR: Training fold $fold failed."
    exit 1
  fi
done

echo "=== All folds trained successfully ==="

# 7) Identify best model configuration (postprocessing, ensembling)
#    This looks at the folds you trained and picks best fold or folds ensemble.
echo "=== Step 3: Determining best configuration/ensemble ==="
apptainer exec --nv \
  --bind /scratch:/scratch \
  --bind /work:/work \
  $CONTAINER_PATH \
  bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate pansegnet_env && \
           nnUNet_find_best_configuration -m 3d_fullres -t $TASK_ID"

if [ $? -ne 0 ]; then
  echo ">>> WARNING: best configuration script encountered issues."
  # Not necessarily fatal, so we won't exit 1
fi

echo "=== Training + configuration selection complete! ==="
echo "Check $RESULTS_FOLDER/nnUNet/3d_fullres/$TASK_NAME for models/folds."
echo "Done."
