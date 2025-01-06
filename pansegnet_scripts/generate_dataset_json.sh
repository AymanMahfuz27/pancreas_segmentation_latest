#!/bin/bash
#SBATCH -J generate_dataset_json
#SBATCH -o generate_dataset_json.o%j
#SBATCH -e generate_dataset_json.e%j
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aymanmahfuz27@utexas.edu

# This script calls your generate_dataset_json.py inside the 'pansegnet_env' conda environment
# within the pansegnet_env.sif container.

PROJECT_DIR=/home1/09999/aymanmahfuz/pancreas_project_latest/pancreas_segmentation_latest
WORK_DIR=/work/09999/aymanmahfuz/ls6
SCRATCH_DIR=/scratch/09999/aymanmahfuz

CONTAINER_PATH=$WORK_DIR/containers/pansegnet_env.sif
PY_SCRIPT=$PROJECT_DIR/generate_dataset_json.py

module purge
module load tacc-apptainer

echo "Starting dataset.json generation..."

# Here we activate conda environment inside the container,
# then run the Python script that calls generate_dataset_json.
apptainer exec --nv $CONTAINER_PATH \
  bash -c "source /opt/conda/etc/profile.d/conda.sh && \
           conda activate pansegnet_env && \
           python $PY_SCRIPT"

if [ $? -ne 0 ]; then
  echo "dataset.json creation failed!"
  exit 1
else
  echo "dataset.json created successfully!"
fi

echo "Done."
