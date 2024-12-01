#!/bin/bash
#SBATCH -J gcp_test
#SBATCH -o gcp_test.o%j
#SBATCH -e gcp_test.e%j
#SBATCH -p gpu-h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00

# Set working directory
SCRATCH_DIR=/scratch/09999/aymanmahfuz
WORK_DIR=/work/09999/aymanmahfuz/ls6

# Load required modules
module purge
module load cuda/12.2

# Initialize conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate medsam2

# Install required packages
echo "Installing required packages..."
pip install google-cloud-storage

# Create test directory
TEST_DIR=$SCRATCH_DIR/pancreas_test_data
mkdir -p $TEST_DIR

# Create a Python script for testing GCP access
cat << 'EOF' > test_gcp_access.py
import os
from google.cloud import storage
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_gcp_access(test_dir):
    try:
        # Verify credentials environment variable
        cred_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        if not cred_path:
            logger.error("GOOGLE_APPLICATION_CREDENTIALS environment variable not set")
            return False
            
        if not os.path.exists(cred_path):
            logger.error(f"Credentials file not found at: {cred_path}")
            return False
            
        logger.info(f"Found credentials file at: {cred_path}")
        
        # Initialize client
        client = storage.Client()
        logger.info("Successfully authenticated with GCP")
        
        # Access bucket
        bucket = client.bucket('pancreas-training-data-dcm')
        logger.info("Successfully accessed bucket")
        
        # List first 5 items in the bucket
        blobs = list(bucket.list_blobs(prefix='pancreas_data/paired/', max_results=5))
        logger.info(f"Found {len(blobs)} items in bucket")
        
        # Print the names of found items
        for blob in blobs:
            logger.info(f"Found item: {blob.name}")
        
        # Download a single case
        if blobs:
            test_case = Path(blobs[0].name).parent.name
            case_dir = Path(test_dir) / test_case
            case_dir.mkdir(parents=True, exist_ok=True)
            
            # Download MRI and mask
            for file_type in ['mri.dcm', 'mask.dcm']:
                blob_path = f"pancreas_data/paired/{test_case}/{file_type}"
                blob = bucket.blob(blob_path)
                local_path = case_dir / file_type
                
                if not blob.exists():
                    logger.error(f"File not found in bucket: {blob_path}")
                    continue
                    
                blob.download_to_filename(str(local_path))
                logger.info(f"Successfully downloaded {file_type} to {local_path}")
                
            return True
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    import sys
    test_dir = sys.argv[1]
    success = test_gcp_access(test_dir)
    sys.exit(0 if success else 1)
EOF

# Verify GCP credentials environment variable
if [ -z "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "Error: GOOGLE_APPLICATION_CREDENTIALS environment variable not set"
    echo "Please set it to the path of your service account key file"
    exit 1
fi

# Run the test script
echo "Testing GCP access and data download..."
python test_gcp_access.py $TEST_DIR

# Check the result
if [ $? -eq 0 ]; then
    echo "GCP test successful"
    echo "Test data downloaded to: $TEST_DIR"
else
    echo "GCP test failed"
    exit 1
fi