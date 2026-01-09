#!/bin/bash
#SBATCH --job-name=jup_an_nb
#SBATCH --mem=32GB
#SBATCH --output=/om2/user/anayebi/code/virtual_zebrafish/zfa/core/slurm_out/%j.out

#SBATCH --qos=yanglab
#SBATCH --partition=yanglab

#SBATCH --gres=gpu:1
#SBATCH --constraint=any-A100
#SBATCH --time=100:00:00

cd /om2/user/anayebi/code/virtual_zebrafish
unset XDG_RUNTIME_DIR
jupyter notebook --ip=0.0.0.0 --port=6000 --no-browser --NotebookApp.token='' --NotebookApp.password=''
