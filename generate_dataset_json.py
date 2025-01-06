#!/usr/bin/env python

"""
create_dataset_json.py

This script uses the `generate_dataset_json` function from nnU-Net
to create a new dataset.json automatically, based on the files in imagesTr/ and imagesTs/.

Usage:
  python create_dataset_json.py
"""

import os
import sys
# We need this from nnU-Net. Some versions place it in different paths,
# but typically it's in nnunet.dataset_conversion.utils or nnunet.utilities.
from typing import Tuple
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *


def get_identifiers_from_splitted_files(folder: str):
    uniques = np.unique([i[:-12] for i in subfiles(folder, suffix='.nii.gz', join=False)])
    return uniques


def generate_dataset_json(output_file: str, imagesTr_dir: str, imagesTs_dir: str, modalities: Tuple,
                          labels: dict, dataset_name: str, sort_keys=True, license: str = "hands off!", dataset_description: str = "",
                          dataset_reference="", dataset_release='0.0'):
    """
    :param output_file: This needs to be the full path to the dataset.json you intend to write, so
    output_file='DATASET_PATH/dataset.json' where the folder DATASET_PATH points to is the one with the
    imagesTr and labelsTr subfolders
    :param imagesTr_dir: path to the imagesTr folder of that dataset
    :param imagesTs_dir: path to the imagesTs folder of that dataset. Can be None
    :param modalities: tuple of strings with modality names. must be in the same order as the images (first entry
    corresponds to _0000.nii.gz, etc). Example: ('T1', 'T2', 'FLAIR').
    :param labels: dict with int->str (key->value) mapping the label IDs to label names. Note that 0 is always
    supposed to be background! Example: {0: 'background', 1: 'edema', 2: 'enhancing tumor'}
    :param dataset_name: The name of the dataset. Can be anything you want
    :param sort_keys: In order to sort or not, the keys in dataset.json
    :param license:
    :param dataset_description:
    :param dataset_reference: website of the dataset, if available
    :param dataset_release:
    :return:
    """
    train_identifiers = get_identifiers_from_splitted_files(imagesTr_dir)

    if imagesTs_dir is not None:
        test_identifiers = get_identifiers_from_splitted_files(imagesTs_dir)
    else:
        test_identifiers = []

    json_dict = {}
    json_dict['name'] = dataset_name
    json_dict['description'] = dataset_description
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = dataset_reference
    json_dict['licence'] = license
    json_dict['release'] = dataset_release
    json_dict['modality'] = {str(i): modalities[i] for i in range(len(modalities))}
    json_dict['labels'] = {str(i): labels[i] for i in labels.keys()}

    json_dict['numTraining'] = len(train_identifiers)
    json_dict['numTest'] = len(test_identifiers)
    json_dict['training'] = [
        {'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i
        in
        train_identifiers]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in test_identifiers]

    if not output_file.endswith("dataset.json"):
        print("WARNING: output file name is not dataset.json! This may be intentional or not. You decide. "
              "Proceeding anyways...")
    save_json(json_dict, os.path.join(output_file), sort_keys=sort_keys)
def main():
    # 1) Define paths
    #    imagesTr_path => e.g. /scratch/.../Task333_T2PancreasMega/imagesTr
    #    imagesTs_path => set to None or path if you have test images
    imagesTr_path = "/scratch/09999/aymanmahfuz/nnUNet_raw_data_base/nnUNet_raw_data/Task333_T2PancreasMega/imagesTr"
    imagesTs_path = None  # if you have no test set; otherwise, provide the folder

    # 2) Output file: dataset.json in the same Task folder
    output_json_path = "/scratch/09999/aymanmahfuz/nnUNet_raw_data_base/nnUNet_raw_data/Task333_T2PancreasMega/dataset.json"

    # 3) Modality definition: for T2 data, we have 1 channel => "0": "MRI" or "T2"
    #    If you had multiple channels, you'd list them in a tuple, e.g. ("T1", "T2"), etc.
    modalities = ("MRI",)

    # 4) Labels: 0 = background, 1 = pancreas
    labels = {
        0: "background",
        1: "pancreas"
    }

    # 5) Additional info for dataset.json
    dataset_name = "T2PancreasMega"
    license_str = "CC-BY-SA 4.0"
    description = "Combined T2 dataset from local + PanSegNet T2"
    reference = "None"
    release = "1.0"

    # 6) Actually call generate_dataset_json
    generate_dataset_json(
        output_file=output_json_path,
        imagesTr_dir=imagesTr_path,
        imagesTs_dir=imagesTs_path,
        modalities=modalities,
        labels=labels,
        dataset_name=dataset_name,
        license=license_str,
        dataset_description=description,
        dataset_reference=reference,
        dataset_release=release,
        sort_keys=True
    )
    print(f"New dataset.json written to {output_json_path}")

if __name__ == "__main__":
    main()
