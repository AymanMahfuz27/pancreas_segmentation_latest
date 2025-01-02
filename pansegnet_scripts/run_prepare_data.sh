#!/bin/bash
#SBATCH -J prep_nnunet_data
#SBATCH -o prep_nnunet_data.o%j
#SBATCH -e prep_nnunet_data.e%j
#SBATCH -p gpu-h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aymanmahfuz27@utexas.edu

# This Bash script executes the prepare_nnunet_data.py script in a container
# on TACC LS6. It will read from /scratch/09999/aymanmahfuz/pancreas_test_data
# and produce processed data in /scratch/09999/aymanmahfuz/pancreas_test_data_preprocessed_paNSegNet

PROJECT_DIR=/home1/09999/aymanmahfuz/pancreas_project_latest/pancreas_segmentation_latest
WORK_DIR=/work/09999/aymanmahfuz/ls6
SCRATCH_DIR=/scratch/09999/aymanmahfuz

CONTAINER_PATH=$WORK_DIR/containers/medsam2.sif
PY_SCRIPT=/project/prepare_nnunet_data.py

module purge
module load tacc-apptainer

echo "Starting data preparation for PaNSegNet/nnU-Net..."

apptainer exec --nv \
    --bind $SCRATCH_DIR:/scratch \
    --bind $PROJECT_DIR:/project \
    --bind $WORK_DIR:/work \
    $CONTAINER_PATH \
    bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate medsam2 && \
             python $PY_SCRIPT"

echo "Data preparation completed. Check the logs above for skipped cases or errors."
