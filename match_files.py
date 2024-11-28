import os
from google.cloud import storage

# Initialize GCP client
client = storage.Client()

# Define bucket names and prefixes
no_mask_bucket_name = 'pancreas_no_mask_files'  # Replace with your actual no-mask bucket name
no_mask_prefix = 'T2w MRI_processed/'
mask_bucket_name = 'pancreas_masked'  # Replace with your actual mask bucket name
mask_prefix = ''  # Assuming masks are at the root of the mask bucket

def extract_key(filename, delimiter='_', num_parts=2):
    """
    Extracts the key based on the first `num_parts` separated by `delimiter`.
    Converts the key to lowercase to ensure case-insensitive matching.
    """
    parts = filename.split(delimiter)
    if len(parts) < num_parts:
        return None
    key = delimiter.join(parts[:num_parts]).lower()
    return key

# List MRI files from the no_mask_bucket
no_mask_bucket = client.bucket(no_mask_bucket_name)
no_mask_blobs = list(no_mask_bucket.list_blobs(prefix=no_mask_prefix))  # Convert iterator to list
mri_files = []
mri_keys = {}
for blob in no_mask_blobs:
    filename = os.path.basename(blob.name)
    if filename.endswith('.DCM'):
        key = extract_key(filename)
        if key:
            mri_files.append(blob.name)
            mri_keys[key] = blob.name

# List Mask files from the mask_bucket
mask_bucket = client.bucket(mask_bucket_name)
mask_blobs = list(mask_bucket.list_blobs(prefix=mask_prefix))  # Convert iterator to list
mask_files = []
mask_keys = {}
for blob in mask_blobs:
    filename = os.path.basename(blob.name)
    if filename.endswith('_mask.dcm'):
        filename_no_suffix = filename.replace('_mask.dcm', '')
        key = extract_key(filename_no_suffix)
        if key:
            mask_files.append(blob.name)
            mask_keys[key] = blob.name

# Find matched keys
matched_keys = set(mri_keys.keys()).intersection(set(mask_keys.keys()))

# Output matched pairs
matched_pairs = [
    (mri_keys[key], mask_keys[key]) 
    for key in matched_keys
]

# Save matched pairs to a file
with open('matched_pairs.txt', 'w') as f:
    for mri, mask in matched_pairs:
        f.write(f"{mri},{mask}\n")

print(f"Total matched pairs: {len(matched_pairs)}")
