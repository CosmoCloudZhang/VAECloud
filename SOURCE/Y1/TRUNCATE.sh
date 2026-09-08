#!/bin/bash
#SBATCH -A m1727
#SBATCH --nodes=1
#SBATCH -q regular
#SBATCH -J SOURCE_Y1_TRUNCATE
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

TAG="Y1"
NAME="SOURCE"
BASE_PATH="/pscratch/sd/y/yhzhang/VAECloud/"
BASE_FOLDER="/global/cfs/cdirs/lsst/groups/MCP/CosmoCloud/VAECloud/"

python -u "${BASE_PATH}/${NAME}/${TAG}/TRUNCATE.py" --tag=$TAG --name=$NAME --folder=$BASE_FOLDER
