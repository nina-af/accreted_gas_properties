#!/bin/bash

#SBATCH -J M2e3_gas_properties
#SBATCH -p vm-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 08:00:00
#SBATCH --mail-user=ninaf@utexas.edu
#SBATCH --mail-type=all
#SBATCH -A AST21048

source $HOME/.bashrc
source $HOME/miniconda3/etc/profile.d/conda.sh

conda init bash
conda activate $HOME/miniconda3/envs/py39

./get_simulation_accreted_gas_properties.py 1>get_gas_properties_2.out 2>get_gas_properties_2.err
