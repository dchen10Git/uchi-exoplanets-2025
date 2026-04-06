#!/bin/bash
#SBATCH --job-name=sim_test
#SBATCH --output=sbatch.out
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --account=pi-fabrycky
#SBATCH --mem-per-cpu=8192

echo Script started.

module load Anaconda3

echo Loaded Anaconda3.

conda activate myenv

echo Environment activated.

pip install -r --dchen10 requirements.txt

echo Requirements installed.

python run_trappist1_sims.py

echo Finished running.
