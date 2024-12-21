#!/bin/bash
#SBATCH -J test_train
#SBATCH -o test_train.o%j
#SBATCH -e test_train.e%j
#SBATCH -p gpu-h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:25:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aymanmahfuz27@utexas.edu

PROJECT_DIR=/home1/09999/aymanmahfuz/pancreas_project_latest/pancreas_segmentation_latest
WORK_DIR=/work/09999/aymanmahfuz/ls6
SCRATCH_DIR=/scratch/09999/aymanmahfuz
CONTAINER_PATH=$WORK_DIR/containers/medsam2.sif

module purge
module load tacc-apptainer

cd $PROJECT_DIR

# Create modify_train.py
cat << 'EOF' > modify_train.py
import fileinput

filename = 'train_3d.py'
search_text = 'loss = function.train_sam(args, net, optimizer, nice_train_loader, epoch)'
replace_text = 'loss = function.train_sam(args, net, optimizer, None, nice_train_loader, epoch)'

with fileinput.FileInput(filename, inplace=True, backup='.bak') as file:
    for line in file:
        print(line.replace(search_text, replace_text), end='')
EOF

# Apply code modification
python modify_train.py

echo "Configuration:"
echo "Project Directory: $PROJECT_DIR"
echo "Work Directory: $WORK_DIR"
echo "Test Data Directory: $SCRATCH_DIR/pancreas_test_data"
echo "Container Path: $CONTAINER_PATH"

echo "Starting single-epoch test training..."

apptainer exec --nv \
    --bind $PROJECT_DIR:/project \
    --bind $WORK_DIR:/work \
    --bind $SCRATCH_DIR:/scratch \
    $CONTAINER_PATH \
    bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate medsam2 && python /project/train_3d.py \
    -net sam2 \
    -exp_name pancreas_test_run \
    -sam_ckpt /work/checkpoints/sam2_hiera_small.pt \
    -sam_config sam2_hiera_t \
    -image_size 1024 \
    -val_freq 1 \
    -prompt bbox \
    -prompt_freq 2 \
    -dataset pancreas \
    -data_path /scratch/pancreas_test_data \
    -b 1 \
    -lr 1e-4 \
    -multimask_output 0 \
    -memory_bank_size 32 \
    -distributed 0"

# Restore original file
mv train_3d.py.bak train_3d.py

if [ $? -eq 0 ]; then
    echo "Test training completed successfully"
else
    echo "Test training failed. Please check the error logs"
    exit 1
fi
