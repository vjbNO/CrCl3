#!/bin/bash
#SBATCH --partition=CPUQ
#SBATCH --account=share-nv-fys
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --time=00:30:00
#SBATCH --job-name=hashpy

module purge
module load Anaconda3/2020.07

python HashMuOverTime_ProcessPoolEx_2Layers_forCluster.py
