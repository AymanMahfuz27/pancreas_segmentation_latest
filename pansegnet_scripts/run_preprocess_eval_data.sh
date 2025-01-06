#!/bin/bash
#SBATCH -J prep_eval_data
#SBATCH -o prep_eval_data.o%j
#SBATCH -e prep_eval_data.e%j
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aymanmahfuz27@utexas.edu

################################################################################
# This SLURM script runs the 'preprocess_eval_data.py' inside the container on TACC.
# It reads from /scratch/09999/aymanmahfuz/pancreas_eval_data/ and converts them
# into /scratch/09999/aymanmahfuz/pancreas_eval_data_preprocessed/
# subdirs: imagesTs, labelsTs
################################################################################

module purge
module load tacc-apptainer

CONTAINER_PATH=/work/09999/aymanmahfuz/ls6/containers/pansegnet_env.sif
PY_SCRIPT=/home1/09999/aymanmahfuz/pancreas_project_latest/pancreas_segmentation_latest/preprocess_eval_data.py

echo "Starting evaluation data preparation..."

apptainer exec --nv $CONTAINER_PATH \
  bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate pansegnet_env && \
           python $PY_SCRIPT"

if [ $? -eq 0 ]; then
  echo "Evaluation data preprocessing completed successfully."
else
  echo "Evaluation data preprocessing failed!"
  exit 1
fi
