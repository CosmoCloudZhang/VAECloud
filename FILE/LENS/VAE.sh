#!/bin/bash
#SBATCH -A m1727
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -q regular
#SBATCH -J FZB_PCA
#SBATCH --time=04:00:00
#SBATCH --mail-type=END
#SBATCH --constraint=gpu
#SBATCH -o LOG/%x_%j.out
#SBATCH --cpus-per-task=128
#SBATCH --mail-user=YunHao.Zhang@ed.ac.uk

# Load modules
module load tensorflow/2.9.0

# Run the application
NAME="LENS"
BASE_PATH="/pscratch/sd/y/yhzhang/VAECloud/"
python -u "${BASE_PATH}/FILE/${NAME}/VAE.py" --path=$BASE_PATH --name=$NAME