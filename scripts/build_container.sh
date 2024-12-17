#!/bin/bash
#SBATCH -J build_container
#SBATCH -o build_container.o%j
#SBATCH -e build_container.e%j
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00

# Set all directory paths
WORK_DIR=/work/09999/aymanmahfuz/ls6
PROJECT_DIR=/home1/09999/aymanmahfuz/pancreas_project_latest/pancreas_segmentation_latest
CONTAINER_BUILD_DIR=$PROJECT_DIR/container_build

# Load required modules
module load tacc-apptainer

# Create container directory if it doesn't exist
mkdir -p $WORK_DIR/containers

# Change to the container build directory
cd $CONTAINER_BUILD_DIR

# Verify files exist
if [ ! -f "medsam2.def" ]; then
    echo "Error: medsam2.def not found in $CONTAINER_BUILD_DIR"
    exit 1
fi

if [ ! -f "environment.yml" ]; then
    echo "Copying environment.yml from project directory..."
    cp $PROJECT_DIR/environment.yml .
fi

echo "Building container from $CONTAINER_BUILD_DIR/medsam2.def"
echo "Current directory: $(pwd)"
echo "Contents of current directory:"
ls -la

# Build the container
apptainer build --fakeroot $WORK_DIR/containers/medsam2.sif medsam2.def

echo "Container built at: $WORK_DIR/containers/medsam2.sif"