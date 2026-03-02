#!/bin/bash
#SBATCH --job-name=sim_test
#SBATCH --output=sbatch.out
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --account=pi-fabrycky
#SBATCH --mem-per-cpu=8192

module load Anaconda3
python run_trappist1_sims.py