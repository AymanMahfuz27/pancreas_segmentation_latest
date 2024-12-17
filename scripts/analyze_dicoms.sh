#!/bin/bash
#SBATCH -J analyze_dicoms
#SBATCH -o analyze_dicoms.o%j
#SBATCH -e analyze_dicoms.e%j
#SBATCH -p gpu-h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00

# Set directories
SCRATCH_DIR=/scratch/09999/aymanmahfuz
WORK_DIR=/work/09999/aymanmahfuz/ls6
PROJECT_DIR=/home1/09999/aymanmahfuz/pancreas_project_latest/pancreas_segmentation_latest

# Change to project directory
cd $PROJECT_DIR

# Load required modules
module purge
module load cuda/12.2

# Initialize conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate medsam2

# Make sure pydicom is installed
pip install pydicom tqdm

# Run the analysis script
python analyze_dicoms.py --data-dir $SCRATCH_DIR/pancreas_test_data

# Save the output to a file as well
python analyze_dicoms.py --data-dir $SCRATCH_DIR/pancreas_test_data > dicom_analysis_results.txt