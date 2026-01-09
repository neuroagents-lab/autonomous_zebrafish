#!/bin/bash
#SBATCH --job-name=bmc_test
#SBATCH --partition=array
#SBATCH --cpus-per-task=25
#SBATCH --mem=512G
#SBATCH --gres=gpu:0
#SBATCH --output=/data/user_data/rdkeller/slurm/out/%A_%a.out
#SBATCH --error=/data/user_data/rdkeller/slurm/err/%A_%a.error
#SBATCH --qos=array_qos
#SBATCH --array=0-24
#SBATCH --mail-type=END
#SBATCH --mail-user=rdkeller@andrew.cmu.edu

source activate /home/rdkeller/miniconda3/envs/fishies
export PYTHONPATH="/home/rdkeller/zebrafish_agent:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

srun python /home/rdkeller/zebrafish_agent/scripts/python_scripts/create_job_info_for_chunking_data.py --processing_chunk_size=5000 --transition="p2a"
srun python /home/rdkeller/zebrafish_agent/zfa/data_comparisons/rdk-controls-percentile-map.py --train-frac=0.5 --num-parallel-jobs=25 --source_cell_type="glial" --target_cell_type="glial" --num-bootstrap-iters=100 --job_ID=$SLURM_ARRAY_TASK_ID --source_animal=1 --target_animal=0