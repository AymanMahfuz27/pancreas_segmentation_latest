#!/bin/bash
#SBATCH -J download_sam2_ckpts
#SBATCH -o download_sam2_ckpts.o%j
#SBATCH -e download_sam2_ckpts.e%j
#SBATCH -p gpu-h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:10:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=YOUR_EMAIL@utexas.edu

# This script downloads the four SAM2 hierarchical checkpoints
# from the official Segment Anything Model 2 location, then places
# them into a local directory on HPC (e.g. /work/<USER>/ls6/checkpoints).

module purge
module load tacc-apptainer

# Define where you want to store the downloaded checkpoints
CHECKPOINT_DIR=/work/09999/aymanmahfuz/ls6/checkpoints  # or wherever you prefer
mkdir -p "$CHECKPOINT_DIR"

# Move into that directory before downloading
cd "$CHECKPOINT_DIR" || {
  echo "Failed to cd into $CHECKPOINT_DIR"
  exit 1
}

# Define the URLs for the checkpoints (same as the original script)
BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/072824/"
sam2_hiera_t_url="${BASE_URL}sam2_hiera_tiny.pt"
sam2_hiera_s_url="${BASE_URL}sam2_hiera_small.pt"
sam2_hiera_b_plus_url="${BASE_URL}sam2_hiera_base_plus.pt"
sam2_hiera_l_url="${BASE_URL}sam2_hiera_large.pt"

# Download each of the four checkpoints using wget
echo "Downloading sam2_hiera_tiny.pt checkpoint..."
wget -O sam2_hiera_tiny.pt "$sam2_hiera_t_url" \
  || { echo "Failed to download checkpoint from $sam2_hiera_t_url"; exit 1; }

echo "Downloading sam2_hiera_small.pt checkpoint..."
wget -O sam2_hiera_small.pt "$sam2_hiera_s_url" \
  || { echo "Failed to download checkpoint from $sam2_hiera_s_url"; exit 1; }

echo "Downloading sam2_hiera_base_plus.pt checkpoint..."
wget -O sam2_hiera_base_plus.pt "$sam2_hiera_b_plus_url" \
  || { echo "Failed to download checkpoint from $sam2_hiera_b_plus_url"; exit 1; }

echo "Downloading sam2_hiera_large.pt checkpoint..."
wget -O sam2_hiera_large.pt "$sam2_hiera_l_url" \
  || { echo "Failed to download checkpoint from $sam2_hiera_l_url"; exit 1; }

echo "All SAM2 hierarchical checkpoints are downloaded successfully to $CHECKPOINT_DIR"
