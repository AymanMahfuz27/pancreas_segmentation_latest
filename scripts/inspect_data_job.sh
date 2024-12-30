#!/bin/bash
#SBATCH -J inspect_data
#SBATCH -o inspect_data.o%j
#SBATCH -e inspect_data.e%j
#SBATCH -p gpu-h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aymanmahfuz27@utexas.edu

PROJECT_DIR=/home1/09999/aymanmahfuz/pancreas_project_latest/pancreas_segmentation_latest
WORK_DIR=/work/09999/aymanmahfuz/ls6
SCRATCH_DIR=/scratch/09999/aymanmahfuz
CONTAINER_PATH=$WORK_DIR/containers/medsam2.sif

module purge
module load tacc-apptainer

cd $PROJECT_DIR

TRAIN_DIR=$SCRATCH_DIR/pancreas_test_data_preproc
EVAL_DIR=$SCRATCH_DIR/pancreas_eval_data_preproc

# We'll create an "inspection_scripts" subdir to store the results
mkdir -p $PROJECT_DIR/inspection_scripts

echo "=== Inspecting TRAIN data in: $TRAIN_DIR ==="
apptainer exec --bind $PROJECT_DIR:/project \
               --bind $SCRATCH_DIR:/scratch \
               $CONTAINER_PATH \
    bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate medsam2 && \
             python /project/inspect_preproc_data.py \
               --data_dir /scratch/pancreas_test_data_preproc \
               --output_path /project/inspection_scripts/train_inspection.json"

echo "=== Inspecting EVAL data in: $EVAL_DIR ==="
apptainer exec --bind $PROJECT_DIR:/project \
               --bind $SCRATCH_DIR:/scratch \
               $CONTAINER_PATH \
    bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate medsam2 && \
             python /project/inspect_preproc_data.py \
               --data_dir /scratch/pancreas_eval_data_preproc \
               --output_path /project/inspection_scripts/eval_inspection.json"

echo "Done. Check the outputs in $PROJECT_DIR/inspection_scripts/ for the JSON results."
