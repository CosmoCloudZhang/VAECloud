#!/bin/bash
#SBATCH -A m1727
#SBATCH --nodes=1
#SBATCH -q regular
#SBATCH -J LENS_PCA
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
module load cray-mpich/8.1.28
module load cray-hdf5-parallel

# Initialize the process
NAME="LENS"
BASE_PATH="/pscratch/sd/y/yhzhang/VAECloud/"
BASE_FOLDER="/global/cfs/cdirs/lsst/groups/PZ/users/CosmoCloud/VAECloud/"

# Run the application
python -u "${BASE_PATH}/FILE/${NAME}/PCA.py" --name=$NAME --folder=$BASE_FOLDER