#!/bin/bash
#SBATCH -J nnunet_bestconf
#SBATCH -o nnunet_bestconf.o%j
#SBATCH -e nnunet_bestconf.e%j
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aymanmahfuz27@utexas.edu

##################################################################################
# After all folds have trained, we run nnUNet_find_best_configuration
# to see which fold or ensemble yields the best dice, and optionally sets up
# the recommended postprocessing.
##################################################################################

module purge
module load tacc-apptainer

export nnUNet_raw_data_base=/scratch/09999/aymanmahfuz/nnUNet_raw_data_base
export nnUNet_preprocessed=/scratch/09999/aymanmahfuz/nnUNet_preprocessed
export RESULTS_FOLDER=/scratch/09999/aymanmahfuz/nnUNet_trained_models

CONTAINER_PATH=/work/09999/aymanmahfuz/ls6/containers/pansegnet_env.sif

TASK_ID=333
echo "=== Determining best configuration for Task ID=$TASK_ID 3d_fullres ==="

apptainer exec --nv \
  --bind /scratch:/scratch \
  --bind /work:/work \
  $CONTAINER_PATH \
  bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate pansegnet_env && \
           nnUNet_find_best_configuration -m 3d_fullres -t $TASK_ID -tr nnTransUNetTrainerV2 -p nnUNetPlansv2.1"

if [ $? -eq 0 ]; then
  echo "Best configuration found!"
else
  echo "Warning: best configuration run had issues."
fi

echo "Done. See $RESULTS_FOLDER/nnUNet/3d_fullres/Task${TASK_ID}_T2PancreasMega for final results."
