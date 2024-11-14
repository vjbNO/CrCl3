#!/bin/bash
#SBATCH --partition=CPUQ
#SBATCH --account=share-nv-fys
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --exclusive
#SBATCH --time=10:00:00
#SBATCH --job-name=vampire
module load OpenMPI/4.1.4-GCC-12.2.0

mpirun --bind-to core --map-by core -np 10 ./vampire-parallel

