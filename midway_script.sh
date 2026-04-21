#!/bin/bash
#SBATCH --job-name=trappist1_sims
#SBATCH --output=sbatch_%a.out
#SBATCH --time=01:00:00
#SBATCH --partition=caslake
#SBATCH --cpus-per-task=28
#SBATCH --account=pi-fabrycky
#SBATCH --mem=56G
#SBATCH --array=15

echo Script started.

module load Anaconda3

echo Loaded Anaconda3.

source activate myenv

echo Environment activated.

/home/dchen10/.conda/envs/myenv/bin/python3 run_trappist1_huang_new.py $SLURM_ARRAY_TASK_ID

echo Finished running.
