#!/bin/bash

#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=128GB
#SBATCH --account=eecs595f23_class
#SBATCH --time=00-08:00:00 

# set up job
module load python cuda
pushd /home/mehars/TitleGenAfterSum
source ./venv/bin/activate

pip install torch
pip install transformers
pip install datasets
pip install tqdm
pip install numpy
pip install evaluate
pip install scikit-learn
pip install peft

# run job
python src/evaluation/human_eval_creation.py
# python src/evaluation/evaluate_subset2.py