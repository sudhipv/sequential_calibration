#!/bin/bash
#SBATCH --account=your-account-name
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=80
#SBATCH --time=00:45:00
##SBATCH --mem-per-cpu=3700M
#SBATCH --job-name=10param_test
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=your-email@abc.com
#SBATCH --mail-type=END


module --force purge --all
module load CCEnv
module load StdEnv/2023

module load python/3.11.5 mpi4py/3.1.6 scipy-stack/2024a


export MPLCONFIGDIR=$SCRATCH

mpiexec -n 160 python -m mpi4py.futures singlephu_8param.py


exit













