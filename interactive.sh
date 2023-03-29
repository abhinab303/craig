#!/bin/bash -lT

# sinteractive --gres=gpu:1
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/aa7514/miniconda3/lib/
source /home/aa7514/miniconda3/etc/profile.d/conda.sh
conda activate sn_coreset

# pip install keras==2.2.5
#SBATCH -n 1
#SBATCH -c 9