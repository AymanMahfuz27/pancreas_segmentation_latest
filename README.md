# Pancreas Segmentation with PaNSegNet & nnU-Net

This repository provides a **containerized HPC pipeline** for **T2-weighted pancreas segmentation** using [PaNSegNet](https://github.com/NUBagciLab/PaNSegNet). The pipeline runs on **TACC’s Lonestar6** (LS6) HPC system via **Apptainer/Singularity** containers, but can be adapted to other HPC platforms.

## Directory Overview

```
.
├── pansegnet_scripts/               # All HPC SLURM scripts for data prep, training, etc.
│   ├── build_container.sh           # Builds container from Docker Hub on TACC (using apptainer build)
│   ├── generate_dataset_json.sh     # Example script to generate dataset.json for nnU-Net
│   ├── nnunet_bestconf.sh           # Finds best nnU-Net config/ensemble after training
│   ├── nnunet_plan.sh               # Runs nnUNet_plan_and_preprocess for your dataset
│   ├── nnunet_train_333.sh          # 5-fold training script for Task333_T2PancreasMega
│   ├── run_evaluation.sh            # SLURM script that executes evaluate_inference.py
│   ├── run_prepare_data.sh          # Prepares training data from DICOM → NIfTI
│   ├── run_preprocess_eval_data.sh  # Prepares evaluation data from DICOM → NIfTI for inference
│   ├── test_container.sh            # Quick test that container + environment are correct
│   └── train_pansegnet.sh           # An alternative script that does plan, train, bestconf in one
├── download_cases.py                # Example code to download DICOM cases from a cloud bucket
├── evaluate_inference.py            # Performs inference + Dice/IoU + single-slice overlay visuals
├── generate_dataset_json.py         # Utility script for auto-creating dataset.json for nnU-Net
├── prepare_nnunet_data.py           # Converts T2 training data from multi-frame DICOM → NIfTI
├── preprocess_eval_data.py          # Converts evaluation data for inference
└── README.md                        # (This file!)
```

## Prerequisites

1. **TACC Lonestar6** (LS6) or another HPC cluster with:
   - Apptainer (Singularity) module
   - GPU nodes (e.g., V100, A100, or H100)
2. **PanSegNet / nnU-Net container** built from Docker or pulled from your Docker Hub image:
   - By default, we build `pansegnet_env.sif` inside `/work/.../containers`.
3. Sufficient time in HPC partitions for:
   - **Plan & Preprocess** (1–2 hrs)
   - **Full 5-fold training** (could be ~24–72 hrs depending on dataset size)
   - **Evaluation** (1–2 hrs)

4. Data:
   - **Training**: T2 multi-frame DICOM volumes in subfolders (each has `mri.dcm`, `mask.dcm`).
   - **Evaluation**: Another set of subfolders to test final model.

## Environment Variables

Throughout the scripts, we rely on:
```bash
export nnUNet_raw_data_base=/scratch/09999/yourusername/nnUNet_raw_data_base
export nnUNet_preprocessed=/scratch/09999/yourusername/nnUNet_preprocessed
export RESULTS_FOLDER=/scratch/09999/yourusername/nnUNet_trained_models
export nnUNet_results=$RESULTS_FOLDER/nnUNet
```
Adjust these in each `.sh` as needed.

## 1. Building (or Using) the Container

1. **build_container.sh**  
   - Submits a SLURM job that runs `apptainer build` from Docker.  
   - Also does a quick container test.  
   ```bash
   sbatch build_container.sh
   ```
   **Alternatively**: If the container is already built in `/work/09999/yourusername/ls6/containers/pansegnet_env.sif`, skip this step.

## 2. Preparing Training Data

1. **Training data** in `/scratch/yourusername/pancreas_test_data` with subfolders:
   ```
   pancreas_test_data/
     ├─ caseXYZ/
     │    ├ mri.dcm
     │    └ mask.dcm
     ├─ ...
   ```
2. Run **`run_prepare_data.sh`**:
   ```bash
   sbatch run_prepare_data.sh
   ```
   This calls `prepare_nnunet_data.py`, converting each subfolder → `imagesTr/CaseID_0000.nii.gz` and `labelsTr/CaseID.nii.gz`. Output in `/scratch/.../pancreas_test_data_preprocessed_paNSegNet`.

## 3. nnU-Net Plan & Preprocess

**nnunet_plan.sh**:
```bash
sbatch nnunet_plan.sh
```
This sets up the standard nnU-Net folder structure in `$nnUNet_preprocessed/Task333_T2PancreasMega/` (or whichever Task ID you used).

## 4. Training (5-Fold)

**nnunet_train_333.sh** or **train_pansegnet.sh**:
```bash
sbatch nnunet_train_333.sh
```
- By default, trains all folds (0..4).  
- If you want fewer folds, edit the for-loop.  
- Once complete, it produces subfolders like `fold_0, fold_1, ... fold_4` in:
  ```
  $RESULTS_FOLDER/nnUNet/3d_fullres/Task333_T2PancreasMega/nnTransUNetTrainerV2__nnUNetPlansv2.1
  ```

## 5. Find Best Model

**nnunet_bestconf.sh**:
```bash
sbatch nnunet_bestconf.sh
```
This runs `nnUNet_find_best_configuration`, picking the best single fold or ensemble. It also sets postprocessing. After success, you’ll see:
```
Best configuration found!
```
and recommended commands for `nnUNet_predict`.

## 6. Preparing Evaluation Data

If you have new T2 DICOM data to *evaluate*:

1. Place subfolders in `/scratch/.../pancreas_eval_data`, each with `mri.dcm`, `mask.dcm`.
2. Run **`run_preprocess_eval_data.sh`** to convert to:
   ```
   pancreas_eval_data_preprocessed/
     ├── imagesTs/ (CaseID_0000.nii.gz)
     └── labelsTs/ (CaseID.nii.gz)
   ```

## 7. Run Inference & Evaluate

Finally, run **`run_evaluation.sh`**:
```bash
sbatch run_evaluation.sh
```
- This calls **`evaluate_inference.py`**, which:
  1. Calls `nnUNet_predict` on `imagesTs/`
  2. Compares predicted masks with `labelsTs/` → prints **Dice & IoU**  
  3. Generates single-slice overlay PNGs in `inference_predictions/visuals/`

Sample log output:
```
Case <X>: Dice=0.82, IoU=0.70
...
Avg Dice: 0.80
```

## 8. Downloading the Model

Your final trained model(s) are in:
```
/scratch/yourusername/nnUNet_trained_models/nnUNet/3d_fullres/Task333_T2PancreasMega/nnTransUNetTrainerV2__nnUNetPlansv2.1
```
**Zip** or **tar** it:
```bash
tar -czvf T2Pancreas_model.tar.gz nnTransUNetTrainerV2__nnUNetPlansv2.1
```
Then **upload** to Box/Git LFS/Drive for your professor. They can place it in `$nnUNet_results/3d_fullres/Task333_T2PancreasMega` to run inference without retraining.

## Troubleshooting

1. **`NoneType` error in `nnUNet_predict`**: Ensure `export nnUNet_results=$RESULTS_FOLDER/nnUNet` is set.
2. **Shape mismatch**: Possibly the mask or MRI was missing slices. Ensure the T2 volumes match in dimension.  
3. **Time Limit**: If folds do not finish in 48 hrs, reduce `--max_epochs` or run each fold in separate SLURM jobs.

---

### In Summary

1. **(Optional)** Build container with `build_container.sh`.  
2. **Prepare** data with `run_prepare_data.sh` → training.  
3. **nnunet_plan.sh** → plan & preprocess.  
4. **nnunet_train_333.sh** → train 5 folds.  
5. **nnunet_bestconf.sh** → pick best model.  
6. **run_preprocess_eval_data.sh** → preprocess new eval data.  
7. **run_evaluation.sh** → inference + Dice/IoU + visuals.
