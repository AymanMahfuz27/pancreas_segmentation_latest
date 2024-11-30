import os
import logging
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from google.cloud import storage
import pydicom
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_processing.log'),
        logging.StreamHandler()
    ]
)

class DataProcessor:
    def __init__(self, destination_path):
        self.client = storage.Client()
        self.bucket = self.client.bucket('pancreas-training-data-dcm')
        self.destination_path = Path(destination_path)
        
    def download_case(self, case_folder):
        """Download a single case (MRI and mask pair) from GCS."""
        try:
            # Create case directory
            case_path = self.destination_path / case_folder
            case_path.mkdir(parents=True, exist_ok=True)
            
            # Download both MRI and mask files
            for file_type in ['mri.dcm', 'mask.dcm']:
                blob_path = f"pancreas_data/paired/{case_folder}/{file_type}"
                blob = self.bucket.blob(blob_path)
                local_path = case_path / file_type
                
                if not local_path.exists():
                    blob.download_to_filename(str(local_path))
                    logging.info(f"Downloaded {blob_path} to {local_path}")
                else:
                    logging.info(f"File already exists: {local_path}")
                    
            # Verify the downloaded pair
            self.verify_case(case_folder)
            
        except Exception as e:
            logging.error(f"Error processing case {case_folder}: {e}")
            return False
        return True

    def verify_case(self, case_folder):
        """Verify that the downloaded MRI and mask files form a valid pair."""
        case_path = self.destination_path / case_folder
        mri_path = case_path / 'mri.dcm'
        mask_path = case_path / 'mask.dcm'
        
        try:
            # Load DICOM files
            mri_dcm = pydicom.dcmread(str(mri_path))
            mask_dcm = pydicom.dcmread(str(mask_path))
            
            # Basic verification checks
            mri_shape = mri_dcm.pixel_array.shape
            mask_shape = mask_dcm.pixel_array.shape
            
            if mri_shape != mask_shape:
                raise ValueError(f"Shape mismatch: MRI {mri_shape} vs Mask {mask_shape}")
            
            # Verify mask is binary
            mask_values = np.unique(mask_dcm.pixel_array)
            if not all(v in [0, 1] for v in mask_values):
                raise ValueError(f"Mask contains non-binary values: {mask_values}")
                
            logging.info(f"Verified case {case_folder}: Shape {mri_shape}, Valid mask values")
            
        except Exception as e:
            logging.error(f"Verification failed for case {case_folder}: {e}")
            return False
        return True

    def process_all_cases(self):
        """Download and verify all cases from the bucket."""
        # List all case folders
        prefix = "pancreas_data/paired/"
        cases = set()
        for blob in self.bucket.list_blobs(prefix=prefix):
            if blob.name.endswith('.dcm'):
                case_folder = Path(blob.name).parent.name
                cases.add(case_folder)
        
        logging.info(f"Found {len(cases)} cases to process")
        
        # Process cases in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for case in cases:
                futures.append(executor.submit(self.download_case, case))
            
            # Track progress with tqdm
            successful = 0
            for future in tqdm(futures, desc="Processing cases"):
                if future.result():
                    successful += 1
                    
        logging.info(f"Successfully processed {successful}/{len(cases)} cases")
        return successful

def main():
    parser = argparse.ArgumentParser(description='Download and verify pancreas MRI data')
    parser.add_argument('--dest', type=str, required=True, 
                      help='Destination path for downloaded data')
    args = parser.parse_args()
    
    processor = DataProcessor(args.dest)
    processor.process_all_cases()

if __name__ == "__main__":
    main()