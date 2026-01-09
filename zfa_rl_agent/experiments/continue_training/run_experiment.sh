#!/bin/bash -l
#SBATCH --job-name=2task
#SBATCH --partition=general
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=76
#SBATCH --mem=256G
#SBATCH --gres=gpu:A6000:1
#SBATCH --output=/data/user_data/rdkeller/rl_training_logs/slurm/out/%x-%j.out
#SBATCH --error=/data/user_data/rdkeller/rl_training_logs/slurm/err/%x-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=rdkeller@andrew.cmu.edu

conda activate fishies
export PYTHONPATH="/home/rdkeller/zebrafish_agent:$PYTHONPATH"
export RAY_DEDUP_LOGS=1
export HYDRA_FULL_ERROR=1
export MUJOCO_GL="egl"
export MUJOCO_EGL_DEVICE_ID=0
## export NUMEXPR_MAX_THREADS=76

TOTAL_TIMESTEPS=30000000
EVAL_FREQ=1000000
CHECKPOINT_FREQ=100000
LEARNING_RATE=0.0003
BATCH_SIZE_MOD=256
N_STEPS=1024
N_EPOCHS=5
USE_FLOW=false
VALUE_COEFF=0.5

FORCE=0.005
IDM_SCALE=1.0
TASK_SCALE=1.0
IDM_TYPE="progress"

NAME="stacked_training_2task-${IDM_TYPE}-2D_drift-${FORCE}"

python /home/rdkeller/zebrafish_agent/zfa_rl_agent/experiments/continue_training/run_experiment.py \
    name="$NAME" \
    total_timesteps="$TOTAL_TIMESTEPS" \
    eval_freq="$EVAL_FREQ" \
    checkpoint_save_freq="$CHECKPOINT_FREQ" \
    batch_mod="$BATCH_SIZE_MOD" \
    learning_rate="$LEARNING_RATE" \
    n_steps="$N_STEPS" \
    n_epochs="$N_EPOCHS" \
    vf_coef="$VALUE_COEFF" \
    force_magnitude="$FORCE" \
    ir_scale="$IDM_SCALE" \
    er_scale="$TASK_SCALE" \
    reward_type="$IDM_TYPE" \

