#!/bin/bash
#SBATCH --job-name=inter-fish-comp-glial-glial
#SBATCH --output=/data/user_data/rdkeller/slurm/out/%A_%a.out
#SBATCH --error=/data/user_data/rdkeller/slurm/err/%A_%a.error
#SBATCH --mem=256G
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --qos=normal

##SBATCH --gres=gpu:0
#SBATCH --time=48:00:00

#SBATCH -c 25
#SBATCH --array=0-24

source activate /home/rdkeller/miniconda3/envs/fishies
which python 

srun python /home/rdkeller/zebrafish_agent/scripts/python_scripts/create_job_info_for_chunking_data.py --processing_chunk_size=5000
srun python /home/rdkeller/zebrafish_agent/zfa/data_comparisons/leo-inter-animal-percentile-map.py --train-frac=0.5 --num-parallel-jobs=25 --source_cell_type="glial" --target_cell_type="glial" --num-bootstrap-iters=100 --job_ID=$SLURM_ARRAY_TASK_ID --source_animal=1 --target_animal=0