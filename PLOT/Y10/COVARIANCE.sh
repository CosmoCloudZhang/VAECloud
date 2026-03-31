#!/bin/bash
#SBATCH -A m1727
#SBATCH --nodes=1
#SBATCH -q regular
#SBATCH --time=04:00:00
#SBATCH --mail-type=END
#SBATCH --constraint=gpu
#SBATCH -o LOG/%x_%j.out
#SBATCH --cpus-per-task=128
#SBATCH --ntasks-per-node=1
#SBATCH -J PLOT_Y10_COVARIANCE
#SBATCH --mail-user=YunHao.Zhang@ed.ac.uk

# Load modules
module load PrgEnv-gnu
module load cray-mpich
module load tensorflow
module load cray-hdf5-parallel

# Initialize the process
TAG="Y10"
BASE_PATH="/pscratch/sd/y/yhzhang/VAECloud/"
BASE_FOLDER="/global/cfs/cdirs/lsst/groups/MCP/CosmoCloud/VAECloud/"

# Run the application
python -u "${BASE_PATH}/PLOT/${TAG}/COVARIANCE.py" --tag=$TAG --folder=$BASE_FOLDER