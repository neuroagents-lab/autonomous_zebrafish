#!/bin/bash
#SBATCH --job-name=inter-fish-comp
#SBATCH --mem=256G
#SBATCH --output=/om2/user/leokoz8/code/virtual_zebrafish/bash_scripts/slurm/outs/%A_%a.out
#SBATCH --error=/dev/null



#SBATCH --qos=normal
#SBATCH --partition=normal

##SBATCH --qos=yanglab
##SBATCH --partition=yanglab

##SBATCH --gres=gpu:0
#SBATCH --time=48:00:00

##SBATCH --nodes=1
#SBATCH -c 25
#SBATCH --array=0-30

source activate /om2/user/leokoz8/envs/zfa
which python 

srun python /om2/user/leokoz8/code/virtual_zebrafish/bash_scripts/python_scripts/create_job_info_for_chunking_data.py --processing_chunk_size=5000
srun python /om2/user/leokoz8/code/virtual_zebrafish/zfa/data_comparisons/leo-inter-animal-percentile-map.py --train-frac=0.5 --num-parallel-jobs=25 --source_cell_type="neural" --target_cell_type="glial" --num-bootstrap-iters=100 --job_ID=$SLURM_ARRAY_TASK_ID --source_animal=1 --target_animal=1