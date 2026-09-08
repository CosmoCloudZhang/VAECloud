#!/bin/bash
#SBATCH -A m1727
#SBATCH --nodes=1
#SBATCH -q regular
#SBATCH --time=04:00:00
#SBATCH --mail-type=END
#SBATCH --constraint=cpu
#SBATCH -o LOG/%x_%j.out
#SBATCH --cpus-per-task=256
#SBATCH --ntasks-per-node=1
#SBATCH -J CELL_Y10_COVARIANCE_GAUSSIAN
#SBATCH --mail-user=YunHao.Zhang@ed.ac.uk

module load conda
source $HOME/.bashrc
conda activate CosmoConda

export NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export PYTHONPATH="/global/homes/y/yhzhang/opt/OneCovariance${PYTHONPATH:+:$PYTHONPATH}"

TAG="Y10"
BASE_PATH="/pscratch/sd/y/yhzhang/VAECloud/"
BASE_FOLDER="/global/cfs/cdirs/lsst/groups/MCP/CosmoCloud/VAECloud/"

srun -u -N 1 -n 1 -c $SLURM_CPUS_PER_TASK python -u /global/homes/y/yhzhang/opt/OneCovariance/covariance.py "${BASE_PATH}CELL/${TAG}/COVARIANCE/GAUSSIAN.ini"
