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
