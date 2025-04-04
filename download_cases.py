import os
import logging
from pathlib import Path
from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def download_case(client, bucket_name, case_name, dest_dir):
    try:
        bucket = client.bucket(bucket_name)
        case_dir = Path(dest_dir) / case_name
        case_dir.mkdir(parents=True, exist_ok=True)

        for file_type in ['mri.dcm', 'mask.dcm']:
            blob_path = f"pancreas_data/paired/{case_name}/{file_type}"
            blob = bucket.blob(blob_path)
            if not blob.exists():
                logger.error(f"File not found: {blob_path}")
                return False
            
            local_path = case_dir / file_type
            blob.download_to_filename(str(local_path))
            logger.info(f"Downloaded {file_type} for case {case_name}")
        
        return True
    except Exception as e:
        logger.error(f"Error downloading case {case_name}: {e}")
        return False

def main(dest_dir, num_cases=None):
    client = storage.Client()
    bucket_name = 'pancreas-training-data-dcm'
    bucket = client.bucket(bucket_name)
    
    # List all cases
    prefix = "pancreas_data/paired/"
    cases = set()
    for blob in bucket.list_blobs(prefix=prefix):
        if blob.name.endswith('mri.dcm'):
            case_name = Path(blob.name).parent.name
            cases.add(case_name)
    
    # cases = sorted(list(cases))[:num_cases]
    logger.info(f"Found {len(cases)} cases to download")
    
    # Download cases in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for case in cases:
            futures.append(
                executor.submit(download_case, client, bucket_name, case, dest_dir)
            )
        
        successful = 0
        for future in tqdm(futures, desc="Downloading cases"):
            if future.result():
                successful += 1
    
    logger.info(f"Successfully downloaded {successful}/{len(cases)} cases")

if __name__ == "__main__":
    import sys
    dest_dir = sys.argv[1]
    main(dest_dir)
