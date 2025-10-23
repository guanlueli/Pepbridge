# **PepBridge: Joint Design of Protein Surface and Backbone Using a Diffusion Bridge Model**

## Overview

**PepBridge** is a novel framework for the joint design of protein surface and backbone structures. It leverages receptor surface geometry and biochemical properties to generate ligand structures that are both conformationally stable and chemically feasible. Starting from a receptor surface represented as a 3D point cloud, PepBridge employs a **Denoising Diffusion Bridge Model (DDBM)** to generate complementary ligand surfaces. A **multi-modal diffusion model** then predicts the corresponding backbone structures, while **Shape-Frame Matching Networks** ensure alignment between the surface geometry and the predicted backbone architecture.

This integrated approach promotes both surface complementarity and structural plausibility in the design of peptide–receptor complexes.

## Installation

We recommend using `miniconda` to create and manage the required environment.

```bash
conda env create -f environment.yml 

conda activate pepbridge

pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu117.html

pip install joblib lmdb easydict wandb

```

## Dataset

### Sources

+ [PepBDB](http://huanglab.phys.hust.edu.cn/pepbdb/db/1cta_A/)
+ [QBioLip](https://yanglab.qd.sdu.edu.cn/Q-BioLiP/Download)

### Directory Structure

After preprocessing, the dataset directory should be organized as follows:

```shell
PepMerge/
├── 1a0n_A/
│   ├── peptide.pdb
│   ├── receptor.pdb
│   ├── pocket.pdb
│   ├── surface_1a0n_A_peptide.pdb.obj
│   └── surface_1a0n_A_pocket.pdb.obj
├── 1a1a_C/
...
Process_Data/
├── names.txt
├── pep_pocket_train_surf_structure_cache.lmdb
└── pep_pocket_test_surf_structure_cache.lmdb           
```
You can use data/pep_dataloader.py to prepare inputs for training.

## Train 

To begin training, run:
 ```
python train_pepbridge.py 
 ```     

Configuration options are defined in config/learn_surf_angle.yaml. Modify this file to customize training settings.


We will add more user-friendly straightforward pipelines (generation and evaluation) later.

## Inference & Generation

To generate new peptide structures:

1. **Run sampling** to generate peptide candidates:
   ```bash
   python inference_pepbridge.py
    ```
2. **Reconstruct full PDB files** from the sampled data:
    ```bash
   python reconstruct.py
    ```

