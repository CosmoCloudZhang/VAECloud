#!/bin/bash
#SBATCH -A m1727
#SBATCH --nodes=1
#SBATCH -q regular
#SBATCH --time=01:00:00
#SBATCH --mail-type=END
#SBATCH --constraint=cpu
#SBATCH -o LOG/%x_%j.out
#SBATCH --cpus-per-task=256
#SBATCH --ntasks-per-node=1
#SBATCH -J CELL_Y1_COVARIANCE_CHECK
#SBATCH --mail-user=YunHao.Zhang@ed.ac.uk

module load conda
source $HOME/.bashrc
conda activate CosmoConda

export HDF5_USE_FILE_LOCKING=FALSE

TAG="Y1"
BASE_PATH="/pscratch/sd/y/yhzhang/VAECloud/"
BASE_FOLDER="/global/cfs/cdirs/lsst/groups/MCP/CosmoCloud/VAECloud/"

python -u "${BASE_PATH}CELL/${TAG}/COVARIANCE/CHECK.py" --tag=$TAG --folder=$BASE_FOLDER
