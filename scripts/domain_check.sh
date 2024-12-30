#!/bin/bash
#SBATCH -J domain_check
#SBATCH -o domain_check.o%j
#SBATCH -e domain_check.e%j
#SBATCH -p gpu-h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00

# Where your script & container are
PROJECT_DIR=/home1/09999/aymanmahfuz/pancreas_project_latest/pancreas_segmentation_latest
CONTAINER_PATH=/work/09999/aymanmahfuz/ls6/containers/medsam2.sif

# The host’s scratch path:
HOST_SCRATCH=/scratch/09999/aymanmahfuz

# Inside-container paths:
TRAIN_DIR_IN_CONTAINER=/scratch/pancreas_test_data
EVAL_DIR_IN_CONTAINER=/scratch/pancreas_eval_data
OUTPUT_JSON_IN_CONTAINER=/scratch/domain_analysis.json

module purge
module load tacc-apptainer

cd $PROJECT_DIR

echo "Running domain check..."

apptainer exec --nv \
    --bind $HOST_SCRATCH:/scratch \
    $CONTAINER_PATH \
    bash -c "source /opt/conda/etc/profile.d/conda.sh && \
             conda activate medsam2 && \
             python domain_check.py \
                --train_dir $TRAIN_DIR_IN_CONTAINER \
                --eval_dir  $EVAL_DIR_IN_CONTAINER \
                --output_json $OUTPUT_JSON_IN_CONTAINER"

echo "Done."
