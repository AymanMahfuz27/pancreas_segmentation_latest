#!/bin/bash
#SBATCH -J build_container
#SBATCH -o build_container.o%j
#SBATCH -e build_container.e%j
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00

# Set directories
WORK_DIR=/work/09999/aymanmahfuz/ls6

# Load required modules
module load tacc-apptainer

# Create container directory if it doesn't exist
mkdir -p $WORK_DIR/containers

# Build directly from Docker Hub
apptainer build $WORK_DIR/containers/medsam2.sif docker://aymanmahfuz/medsam2:latest

echo "Container built at: $WORK_DIR/containers/medsam2.sif"