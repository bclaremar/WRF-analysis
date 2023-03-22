#!/bin/bash
##SBATCH -t 00:01:00
#SBATCH -n 3 
#SBATCH -N 3
#SBATCH -c 4
#SBATCH -w node0[1-3]
##SBATCH --tasks-per-node=3
#SBATCH -J wrf
#SBATCH -o wrf_%j.out
#SBATCH -e wrf_%j.out
#SBATCH -W
#ml gnu
#ml gnu/8.3.0/openmpi/3.1.3

cd $SLURM_SUBMIT_DIR
#cat $0
#export OMP_NUM_THREADS=4
nt=1
if [ -n "$SLURM_CPUS_PER_TASK" ]; then
  nt=$SLURM_CPUS_PER_TASK
fi
export OMP_NUM_THREADS=$nt
export I_MPI_PIN_DOMAIN=omp

mpiexec -n 3 --mca mpi_preconnect_all 1 ./wrf.exe

