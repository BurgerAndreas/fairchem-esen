#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --job-name=esen_training
#SBATCH --output=/scratch/aburger/outslurm/slurm-%j.txt
#SBATCH --error=/scratch/aburger/outslurm/slurm-%j.txt

# option	short option	meaning	notes
# # --nodes	-N	number of nodes	Recommended to always include this
# # --ntasks-per-node		number of tasks for srun/mpirun to launch per node	Prefer this over --ntasks
# # --ntasks	-n	number of tasks for srun/mpirun to launch	
# # --cpus-per-task	-c	number of cores per task;	Typically for (OpenMP) threads
# # --time	-t	duration of the job	
# # --job-name	-J	specify a name for the job	
# # --output	-o	file to redirect standard ouput to	Can be a pattern using e.g. %j for the jobid.
# # --mail-type		when to send email (e.g. BEGIN, END, FAIL, ALL)
# # --gpus-per-node		number of gpus to use on each node	Either 1 or 4 is allowed on the GPU subcluster
# # --partition	-p	partition to submit to	See below for available partitions
# # --account	-A	slurm account to use	For many users, this is automatic on Trillium
# # --mem		amount of memory requested	Ignored on Trillium, you get all the memory

# 252 GPUs provided by 63 GPU compute nodes.
# Each GPU compute node has 4 NVIDIA H100 (SXM) GPUs with 80 GB of dedicated VRAM
# Each GPU compute node also has 96 cores from one 96-core AMD EPYC 9654 CPUs ("Zen 4" a.k.a. "Genoa")

# Min. size of jobs: 1/4 node (24 cores / 1GPU)
# Max. walltime: 24 hours

# get environment variables
source .env

# cd $SLURM_SUBMIT_DIR
# cd /project/rrg-aspuru/aburger/fairchem-esen
cd /scratch/aburger/fairchem-esen

# load modules
module load StdEnv/2023  gcc/12.3 cuda/12.6
# module load StdEnv/2023
# module load openmpi/4.1.5
# module load python/3.11.5

# Set cache directories to writable locations (avoid permission issues on cluster)
export UV_CACHE_DIR=/scratch/aburger/.cache/uv
export MPLCONFIGDIR=/scratch/aburger/.cache/matplotlib
export XDG_CACHE_HOME=/scratch/aburger/.cache
export FAIRCHEM_CACHE_DIR=/scratch/aburger/.cache/fairchem
export WANDB_CACHE_DIR=/scratch/aburger/wandb
export WANDB_CONFIG_DIR=/scratch/aburger/wandb
export WANDB_DIR=/scratch/aburger/wandb
mkdir -p "$UV_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$FAIRCHEM_CACHE_DIR" "$WANDB_CACHE_DIR" "$WANDB_DIR"

echo "Activating Python environment"
# Activate Python environment 
source .venv/bin/activate

# Check GPU allocation
# srun nvidia-smi

# Run your workload
srun uv run "$@"