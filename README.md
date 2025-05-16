# PepBridge: Joint Design of Protein Surface and Backbone Using a Diffusion Bridge Model


### Environment


```bash
conda env create -f environment.yml 

conda activate surf_se3

pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu117.html

pip install joblib lmdb easydict

```

### Train 

 ```
python train.py 
 ```

### Data and Weights Download

+ PepMerge_release.zip: contains filtered data of peptide-receptor pairs. You can also download [PepBDB](http://huanglab.phys.hust.edu.cn/pepbdb/db/1cta_A/) and [QBioLip](https://yanglab.qd.sdu.edu.cn/Q-BioLiP/Download), and use ```playgrounds/gen_dataset```.ipynb to reproduce the dataset.
+ pep_pocket_test_surf_structure_cache.lmdb: test dataset
+ pep_pocket_train_surf_structure_cache.lmdb: train dataset
+ model1.pt, model2.pt: trained models 

## Usage

### Inference and Generate

```models_con/inference.py``` to reconstruct PDB files and use ```models_con/sample.py``` to sample.
