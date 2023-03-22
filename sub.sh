#!/bin/bash
##SBATCH --nodes=1
##SBATCH --ntasks-per-node=4
##SBATCH --nodelist=node02

cd $SLURM_SUBMIT_DIR

#ls
#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun:-n 8 ./wrf.exe
