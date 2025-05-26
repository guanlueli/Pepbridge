# PepBridge: Joint Design of Protein Surface and Backbone Using a Diffusion Bridge Model

## Description

Implementation for "PepBridge: Joint Design of Protein Surface and Backbone Using a Diffusion Bridge Model". In this work, we introduce PepBridge, a novel framework for the joint design of protein surface and structure that seamlessly integrates receptor surface geometry and biochemical properties. Starting with a receptor surface represented as a 3D point cloud, PepBridge generates complete protein structures through a multi-step process. First, it employs denoising diffusion bridge models (DDBMs) to map receptor surfaces to ligand surfaces. Next, a multi-model diffusion model predicts the corresponding structure, while Shape-Frame Matching Networks ensure alignment between surface geometry and backbone architecture. This integrated approach facilitates surface complementarity, conformational stability, and chemical feasibility. 

## Installation

We use miniconda to install a conda environment with necessary dependencies.

```bash
conda env create -f environment.yml 

conda activate surf_se3

pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu117.html

pip install joblib lmdb easydict

```

### Train 

 ```
python train_pepbridge.py 
 ```

### Data and Weights

+ PepMerge_release.zip: contains filtered data of peptide-receptor pairs. You can also download [PepBDB](http://huanglab.phys.hust.edu.cn/pepbdb/db/1cta_A/) and [QBioLip](https://yanglab.qd.sdu.edu.cn/Q-BioLiP/Download), and use ```data_processing/gen_dataset.ipynb``` to reproduce the dataset. 
Navigate down to "Download Protocols" and follow the instructions depending on your location.
Then run the following with <path_pdb_dir> replaced with the location of PDB.
```
python data/pep_dataloader.py
```


## Usage

We will add more user-friendly straightforward pipelines (generation and evaluation) later.

### Inference and Generate

```models_con/inference.py``` to reconstruct PDB files and use ```models_con/sample.py``` to sample.
