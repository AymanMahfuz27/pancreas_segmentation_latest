#!/bin/bash
#SBATCH -J test_container
#SBATCH -o test_container.o%j
#SBATCH -e test_container.e%j
#SBATCH -p gpu-h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00

module load tacc-apptainer

CONTAINER=/work/09999/aymanmahfuz/ls6/containers/pansegnet_env.sif

apptainer exec --nv $CONTAINER \
  bash -c "source /opt/conda/etc/profile.d/conda.sh && \
           conda activate pansegnet_env && \
           python -c 'import nnunet; print(\"OK!\")'"
