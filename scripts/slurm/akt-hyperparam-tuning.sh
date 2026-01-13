#!/bin/bash
#SBATCH --job-name=64_hypertune
#SBATCH --output=/data/user_data/akirscht/slurm/out/%A_%a_avg_flows.out
#SBATCH --error=/data/user_data/akirscht/slurm/err/%A_%a_avg_flows.error
#SBATCH --partition=general
#SBATCH --nodelist=babel-7-1 
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00

source ~/miniconda3/etc/profile.d/conda.sh
# Activate the specific Conda environment
conda activate zfa

#

# Define the source directories
SOURCE_MVK_FRAME_DIR="/scratch/$USER/marine_video_kit_50"
SOURCE_MVK_FLOW_DIR="/scratch/$USER/marine_video_kit_50/processed/MarineVideoKit_flows"

python ~/zebrafish_agent/scripts/python_scripts/akt-hyperparam-tuning.py
