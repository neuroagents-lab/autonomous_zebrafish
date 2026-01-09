#!/bin/bash
#SBATCH --job-name=jup_nb
#SBATCH --mem=128GB
#SBATCH --output=/om2/user/leokoz8/code/virtual_zebrafish/bash_scripts/slurm/outs/%j.out
#SBATCH --error=/om2/user/leokoz8/code/virtual_zebrafish/bash_scripts/slurm/errs/%j.error


#SBATCH --qos=normal
#SBATCH --partition=normal

##SBATCH --qos=yanglab
##SBATCH --partition=yanglab

#SBATCH --gres=gpu:0
#SBATCH --time=0-12:00:00


cd /om2/user/leokoz8/code/virtual_zebrafish
unset XDG_RUNTIME_DIR
source activate /om2/user/leokoz8/envs/$1
#jupyter notebook --port=6789 --no-browser --NotebookApp.allow_origin='*' --NotebookApp.ip='0.0.0.0'
jupyter notebook --ip=0.0.0.0 --port=6876 --no-browser --NotebookApp.allow_origin='*' --NotebookApp.port_retries=0 