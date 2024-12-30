#!/bin/bash
#SBATCH -J preproc_pancreas
#SBATCH -o preproc_pancreas.o%j
#SBATCH -e preproc_pancreas.e%j
#SBATCH -p gpu-h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aymanmahfuz27@utexas.edu

# Paths
PROJECT_DIR=/home1/09999/aymanmahfuz/pancreas_project_latest/pancreas_segmentation_latest
WORK_DIR=/work/09999/aymanmahfuz/ls6
SCRATCH_DIR=/scratch/09999/aymanmahfuz
CONTAINER_PATH=$WORK_DIR/containers/medsam2.sif

# Input data
TRAIN_INPUT_DIR=$SCRATCH_DIR/pancreas_test_data
EVAL_INPUT_DIR=$SCRATCH_DIR/pancreas_eval_data

# Output directories for preprocessed data
TRAIN_OUTPUT_DIR=$SCRATCH_DIR/pancreas_test_data_preproc
EVAL_OUTPUT_DIR=$SCRATCH_DIR/pancreas_eval_data_preproc

module purge
module load tacc-apptainer

echo "=== Preprocessing Script ==="
echo "Project Dir:   $PROJECT_DIR"
echo "Container:     $CONTAINER_PATH"
echo "Train Input:   $TRAIN_INPUT_DIR"
echo "Train Output:  $TRAIN_OUTPUT_DIR"
echo "Eval Input:    $EVAL_INPUT_DIR"
echo "Eval Output:   $EVAL_OUTPUT_DIR"

# Make sure the python script is in the project directory, or provide the full path
PREPROCESS_SCRIPT=$PROJECT_DIR/preprocess_data.py

# 1) Preprocess Training Data
echo "=== Preprocessing TRAINING Data ==="
apptainer exec --nv \
  --bind $SCRATCH_DIR:/scratch \
  --bind $WORK_DIR:/work \
  --bind $PROJECT_DIR:/project \
  $CONTAINER_PATH \
  bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate medsam2 && \
           python /project/preprocess_data.py \
             --input_dir  /scratch/pancreas_test_data \
             --output_dir /scratch/pancreas_test_data_preproc \
             --low_percent 0.5 \
             --high_percent 99.5 \
             --target_size 1024"

# 2) Preprocess Evaluation Data
echo "=== Preprocessing EVALUATION Data ==="
apptainer exec --nv \
  --bind $SCRATCH_DIR:/scratch \
  --bind $WORK_DIR:/work \
  --bind $PROJECT_DIR:/project \
  $CONTAINER_PATH \
  bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate medsam2 && \
           python /project/preprocess_data.py \
             --input_dir  /scratch/pancreas_eval_data \
             --output_dir /scratch/pancreas_eval_data_preproc \
             --low_percent 0.5 \
             --high_percent 99.5 \
             --target_size 1024"

echo "[ALL DONE] Check output in $TRAIN_OUTPUT_DIR and $EVAL_OUTPUT_DIR"
