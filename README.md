# Semantic Change Detection with Graph Neural Networks
Codebase for CSC2611H(Computational Models of Semantic Change) Final Project

### Python environment setup with Conda

```bash
conda create -n semgnn python=3.9
conda activate semgnn

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
conda install pyg -c pyg==2.2.0

conda install -c anaconda numpy==1.23.3
conda install -c anaconda pandas==1.4.3
conda install -c anaconda scipy==1.9.3
conda install -c anaconda scikit-learn==1.1.3

conda install -c conda-forge pyarrow==8.0.0
conda install -c conda-forge tqdm==4.64.1

pip install transformers==4.24.0
```

### Running SemGNN
```bash

conda activate semgnn

# SemGNN pipeline include following steps:

# Running preprocessing for each corpus
python3 preprocessing.py --file_id 1
python3 preprocessing.py --file_id 2

# Model training for each corpus
python3 train_eval.py --file_id 1
python3 train_eval.py --file_id 2

# Post-processing + results
python3 post_processing.py
```
