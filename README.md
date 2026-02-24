# VFM_CatFlow
Code for the paper "Variational Flow Matching for Graph Generation"

## Installation
* Create the environment: ```conda create -c conda-forge -n VFM rdkit=2023.03.2 python=3.9```
* Activate the environment: ```conda activate VFM```
* Verify the installation of rdkit:
  * ```python -c 'from rdkit import Chem'```
* Install the nvcc drivers: ```conda install -c "nvidia/label/cuda-12.1.0" cuda```
* Install Pytorch: ```(python -m) pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121```
* Install PyG related packages: ```(python -m) pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html```
* Please ensure the synchronization of the versions of *nvcc drivers, Pytorch and PyG*!
* Install pytorch_warmup, torch_geometric, torchdiffeq and torch-ema: ```(python -m) pip install pytorch_warmup torch_geometric torchdiffeq torch-ema```
* Install numpy, scipy, pandas, scikit-learn, tqdm, wandb, accelerate, pyemd, dill and fcd: ```(python -m) pip install numpy scipy pandas scikit-learn tqdm wandb accelerate pyemd dill fcd```
* Install orca
  * ```cd evaluation```
  * ```python -m pip install -e .```

## Usage
```python main.py```
