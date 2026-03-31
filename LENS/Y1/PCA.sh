#!/bin/bash
#SBATCH -A m1727
#SBATCH --nodes=1
#SBATCH -q regular
#SBATCH -J LENS_Y1_PCA
#SBATCH --time=04:00:00
#SBATCH --mail-type=END
#SBATCH --constraint=gpu
#SBATCH -o LOG/%x_%j.out
#SBATCH --cpus-per-task=128
#SBATCH --ntasks-per-node=1
#SBATCH --mail-user=YunHao.Zhang@ed.ac.uk

# Load modules
module load cray-mpich
module load PrgEnv-gnu
module load tensorflow
module load cray-hdf5-parallel

# Initialize the process
TAG="Y1"
NAME="LENS"
BASE_PATH="/pscratch/sd/y/yhzhang/VAECloud/"
BASE_FOLDER="/global/cfs/cdirs/lsst/groups/MCP/CosmoCloud/VAECloud/"

# Run the application
python -u "${BASE_PATH}/${NAME}/${TAG}/PCA.py" --tag=$TAG --name=$NAME --folder=$BASE_FOLDER