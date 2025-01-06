#!/bin/bash
#SBATCH -J build_container
#SBATCH -o build_container.o%j
#SBATCH -e build_container.e%j
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00

# Load Apptainer (Singularity) module
module load tacc-apptainer

# Define your container path on TACC
WORK_DIR=/work/09999/aymanmahfuz/ls6
CONTAINER_NAME=pansegnet_env.sif
mkdir -p $WORK_DIR/containers

# Build from Docker Hub
apptainer build $WORK_DIR/containers/$CONTAINER_NAME \
    docker://aymanmahfuz/pansegnet_env:latest

echo "Container built at: $WORK_DIR/containers/$CONTAINER_NAME"


#Test out the container

echo "Testing container..."
module load tacc-apptainer
apptainer exec --nv /work/09999/aymanmahfuz/ls6/containers/pansegnet_env.sif \
    python -c "import nnunet; import pansegnet; print('Success!')"

if [ $? -eq 0 ]; then
    echo "Container test successful!"
else
    echo "Container test failed!"
fi