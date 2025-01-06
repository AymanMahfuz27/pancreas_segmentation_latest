#!/bin/bash
#SBATCH -J eval_inference
#SBATCH -o eval_inference.o%j
#SBATCH -e eval_inference.e%j
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aymanmahfuz27@utexas.edu

################################################################################
# This SLURM script executes "evaluate_inference.py":
#  1) Runs nnUNet_predict on /scratch/09999/aymanmahfuz/pancreas_eval_data_preprocessed/imagesTs
#  2) Compares predicted masks with /labelsTs, calculates Dice/IoU
#  3) Creates side-by-side visuals for each case
#  4) Summarizes performance
################################################################################

module purge
module load tacc-apptainer


export nnUNet_raw_data_base=/scratch/09999/aymanmahfuz/nnUNet_raw_data_base
export nnUNet_preprocessed=/scratch/09999/aymanmahfuz/nnUNet_preprocessed
export RESULTS_FOLDER=/scratch/09999/aymanmahfuz/nnUNet_trained_models
export nnUNet_results=$RESULTS_FOLDER/nnUNet

CONTAINER_PATH=/work/09999/aymanmahfuz/ls6/containers/pansegnet_env.sif
EVAL_PY=/home1/09999/aymanmahfuz/pancreas_project_latest/pancreas_segmentation_latest/evaluate_inference.py

echo "Starting evaluation (inference + metrics + visuals)..."

apptainer exec --nv $CONTAINER_PATH \
bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate pansegnet_env && \
         python $EVAL_PY"

if [ $? -eq 0 ]; then
  echo "Evaluation script completed successfully."
else
  echo "Evaluation script failed!"
  exit 1
fi
