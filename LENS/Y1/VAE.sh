#!/bin/bash
#SBATCH -A m1727
#SBATCH --nodes=4
#SBATCH -q regular
#SBATCH -J LENS_Y1_VAE
#SBATCH --time=04:00:00
#SBATCH --mail-type=END
#SBATCH --constraint=gpu
#SBATCH -o LOG/%x_%j.out
#SBATCH --cpus-per-task=128
#SBATCH --ntasks-per-node=1
#SBATCH --mail-user=YunHao.Zhang@ed.ac.uk

# Load modules
module load tensorflow
module load PrgEnv-gnu
module load cray-mpich/8.1.30
module load cray-hdf5-parallel

# Initialize the process
TAG="Y1"
NAME="LENS"
BASE_PATH="/pscratch/sd/y/yhzhang/VAECloud/"
BASE_FOLDER="/global/cfs/cdirs/lsst/groups/MCP/CosmoCloud/VAECloud/"

# Run the application
python -u "${BASE_PATH}/FILE/${NAME}/${TAG}/VAE.py" --tag=$TAG --name=$NAME --folder=$BASE_FOLDER