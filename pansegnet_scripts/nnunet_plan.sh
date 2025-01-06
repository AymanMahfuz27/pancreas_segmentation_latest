#!/bin/bash
#SBATCH -J nnunet_plan
#SBATCH -o nnunet_plan.o%j
#SBATCH -e nnunet_plan.e%j
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aymanmahfuz27@utexas.edu

##################################################################################
# Plan & Preprocess (without verify_dataset_integrity) for Task 333 T2 data.
# We'll do only this step so we don't risk hitting time limits while training.
#
# After this finishes, check the logs and the $nnUNet_preprocessed folder to see
# if everything processed correctly. Then run Script B for training.
##################################################################################

module purge
module load tacc-apptainer

# Export environment variables
export nnUNet_raw_data_base=/scratch/09999/aymanmahfuz/nnUNet_raw_data_base
export nnUNet_preprocessed=/scratch/09999/aymanmahfuz/nnUNet_preprocessed
export RESULTS_FOLDER=/scratch/09999/aymanmahfuz/nnUNet_trained_models

CONTAINER_PATH=/work/09999/aymanmahfuz/ls6/containers/pansegnet_env.sif

TASK_ID=333
TASK_NAME="Task${TASK_ID}_T2PancreasMega"

echo "=== Running nnUNet_plan_and_preprocess for $TASK_NAME (no verify) ==="

apptainer exec --nv \
  --bind /scratch:/scratch \
  --bind /work:/work \
  $CONTAINER_PATH \
  bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate pansegnet_env && \
           nnUNet_plan_and_preprocess -t $TASK_ID"

if [ $? -eq 0 ]; then
    echo "Plan & Preprocess completed successfully."
else
    echo "Plan & Preprocess failed!"
    exit 1
fi
