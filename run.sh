#!/bin/bash

#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=32GB
#SBATCH --account=eecs595f23_class

# set up job
module load python cuda
pushd /home/mehars/TitleGenAfterSum/src
source ../venv/bin/activate

pip install torch
pip install transformers
pip install datasets
pip install tqdm
pip install numpy
pip install evaluate
pip install scikit-learn

# run job
python src/train.py
# python src/explore.py