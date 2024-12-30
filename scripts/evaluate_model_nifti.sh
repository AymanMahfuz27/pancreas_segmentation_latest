#!/bin/bash
#SBATCH -J eval_nifti
#SBATCH -o eval_nifti.o%j
#SBATCH -e eval_nifti.e%j
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

cd "$PROJECT_DIR"

# 1) Path to your best model:
BEST_MODEL_PATH="$PROJECT_DIR/logs/pancreas_full_training_2024_12_25_18_03_32/Model/best_model.pth"

# 2) Preprocessed NIfTI evaluation data directory
EVAL_DATA_DIR="/scratch/09999/aymanmahfuz/pancreas_eval_data_preproc"

# 3) Output directory for metrics & optional overlays
OUTPUT_DIR="/scratch/09999/aymanmahfuz/eval_nifti_results"
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"


echo "Now running evaluation on preprocessed NIfTI data..."

apptainer exec --nv \
  --bind "$SCRATCH_DIR:/scratch" \
  --bind "$PROJECT_DIR:/project" \
  --bind "/work/09999/aymanmahfuz/ls6/checkpoints:/checkpoints" \
  --bind "$EVAL_DATA_DIR:/eval_data" \
  "$CONTAINER_PATH" \
  bash -c "
    source /opt/conda/etc/profile.d/conda.sh && conda activate medsam2 && \
    python run_evaluation_on_nifti_data.py \
      --model_path '$BEST_MODEL_PATH' \
      --data_dir '/eval_data' \
      --output_dir '$OUTPUT_DIR'
  "

echo "Evaluation completed. Check $OUTPUT_DIR for results and overlays!"
