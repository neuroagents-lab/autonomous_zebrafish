#!/bin/bash
#
#set the job name (output file and error file)
#SBATCH --job-name=extract_val_frames
#SBATCH --output=/om2/user/anayebi/code/virtual_zebrafish/zfa/core/slurm_out/extract_val_frames_%A_%a.out
#SBATCH --error=/om2/user/anayebi/code/virtual_zebrafish/zfa/core/slurm_err/extract_val_frames_%A_%a.err
#################
#set time needed
#SBATCH --time=12:00:00
#################
#SBATCH -p yanglab
#################
#SBATCH --nodes=1
#SBATCH -c 20
#################
#SBATCH --mem=256000
#################
#SBATCH --array=0

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

srun /om2/user/anayebi/miniconda/envs/pong_env/bin/python ${HOME}/zebrafish_agent/scripts/dataset_dl/videos_to_frames.py --mode="val"