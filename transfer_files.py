import os
import sys
import logging
from google.cloud import storage
from google.api_core.exceptions import GoogleAPIError

logging.basicConfig(
    filename='file_transfer.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def initialize_storage_client():
    try:
        client = storage.Client()
        logging.info("Initialized Google Cloud Storage client.")
        return client
    except Exception as e:
        logging.error(f"Failed to initialize GCP Storage client: {e}")
        sys.exit(1)

def read_matched_pairs(file_path):
    matched_pairs = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) != 2:
                    logging.warning(f"Invalid line format (skipped): {line.strip()}")
                    continue
                mri_path = parts[0].strip()
                mask_path = parts[1].strip()
                matched_pairs.append((mri_path, mask_path))
        logging.info(f"Read {len(matched_pairs)} matched pairs from {file_path}.")
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        sys.exit(1)
    return matched_pairs

def copy_blob(client, source_bucket_name, source_blob_name, destination_bucket_name, destination_blob_name):
    try:
        source_bucket = client.bucket(source_bucket_name)
        source_blob = source_bucket.blob(source_blob_name)
        destination_bucket = client.bucket(destination_bucket_name)

        if not source_blob.exists():
            logging.error(f"Source blob does not exist: gs://{source_bucket_name}/{source_blob_name}")
            return False

        # Create destination blob
        destination_blob = destination_bucket.blob(destination_blob_name)
        
        # Copy using rewrite
        token = None
        while True:
            token, bytes_rewritten, total_bytes = destination_blob.rewrite(
                source=source_blob, token=token)
            if token is None:
                break

        logging.info(f"Copied gs://{source_bucket_name}/{source_blob_name} to gs://{destination_bucket_name}/{destination_blob_name}")
        return True
        
    except GoogleAPIError as e:
        logging.error(f"Failed to copy {source_blob_name}: {e}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error during copy: {e}")
        return False

def main():
    # Configuration
    matched_pairs_file = 'matched_pairs.txt'
    destination_bucket_name = 'pancreas-training-data-dcm'
    destination_prefix = 'pancreas_data/paired'

    # Initialize client
    client = initialize_storage_client()

    # Read matched pairs
    matched_pairs = read_matched_pairs(matched_pairs_file)

    if not matched_pairs:
        logging.warning("No matched pairs to transfer. Exiting.")
        sys.exit(0)

    # Counter for successful transfers
    successful_transfers = 0
    total_pairs = len(matched_pairs)

    # Iterate over each pair and copy files
    for i, (mri_path, mask_path) in enumerate(matched_pairs, 1):
        # Extract filename from MRI path for organizing in destination
        filename = os.path.splitext(os.path.basename(mri_path))[0].split('_WIP')[0]
        
        # Define destination paths
        mri_destination = f"{destination_prefix}/{filename}/mri.dcm"
        mask_destination = f"{destination_prefix}/{filename}/mask.dcm"

        logging.info(f"Processing pair {i}/{total_pairs}: {filename}")

        # Copy MRI file
        mri_success = copy_blob(
            client, 
            'pancreas_no_mask_files',
            mri_path,
            destination_bucket_name,
            mri_destination
        )

        # Copy Mask file
        mask_success = copy_blob(
            client,
            'pancreas_masked',
            mask_path,
            destination_bucket_name,
            mask_destination
        )

        if mri_success and mask_success:
            successful_transfers += 1
            logging.info(f"Successfully transferred pair {successful_transfers}/{total_pairs}: {filename}")
        else:
            logging.error(f"Failed to transfer one or both files for {filename}")

    logging.info(f"File transfer completed. Successfully transferred {successful_transfers}/{total_pairs} pairs.")

if __name__ == "__main__":
    main()