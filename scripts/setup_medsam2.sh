#setup_medsam2.sh
#description: Setup the MedSAM2 environment.
#!/bin/bash
#SBATCH -J medsam2_setup    # Job name
#SBATCH -o medsam2.o%j      # Name of stdout output file
#SBATCH -e medsam2.e%j      # Name of stderr error file
#SBATCH -p gpu-a100         # Queue name
#SBATCH -N 1                # Total number of nodes
#SBATCH -n 1                # Total number of mpi tasks
#SBATCH -t 02:00:00        # Run time (hh:mm:ss)

# Set conda environment directory
export CONDA_ENVS_PATH=/work/09999/aymanmahfuz/ls6/conda_envs

# Load required modules
module purge
module load cuda/12.2

# Initialize conda
source ~/miniconda3/etc/profile.d/conda.sh

# Create environment
cd /work/09999/aymanmahfuz/ls6
conda env create -f /home1/09999/aymanmahfuz/pancreas_project_latest/pancreas_segmentation_latest/environment.yml