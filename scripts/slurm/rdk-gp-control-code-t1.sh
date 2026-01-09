#!/bin/bash
#SBATCH --job-name=ctrls
#SBATCH --output=/data/user_data/rdkeller/slurm/out/%A_%a.out
#SBATCH --error=/data/user_data/rdkeller/slurm/err/%A_%a.error
#SBATCH --mem=512G
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --qos=normal

##SBATCH --gres=gpu:0
#SBATCH --time=48:00:00

#SBATCH -c 25
#SBATCH --array=0-24

source activate /home/rdkeller/miniconda3/envs/fishies
which python 

## source animal key || 0-1: fish, 2: white noise, 3: gaussian process using fish 0, 4: gaussian process using fish 1, 5: bang-bang controller

srun python /home/rdkeller/zebrafish_agent/scripts/python_scripts/create_job_info_for_chunking_data.py --processing_chunk_size=5000
srun python /home/rdkeller/zebrafish_agent/zfa/data_comparisons/rdk-controls-percentile-map.py --train-frac=0.5 --num-parallel-jobs=25 --source_cell_type="glial" --target_cell_type="glial" --num-bootstrap-iters=100 --job_ID=$SLURM_ARRAY_TASK_ID --source_animal=4 --target_animal=0