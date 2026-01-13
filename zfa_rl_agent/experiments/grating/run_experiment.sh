#!/bin/bash -l
#SBATCH --job-name=grating
#SBATCH --partition=general
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=76
#SBATCH --mem=64G
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
export WANDB_DISABLE_CACHE=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
##export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.6,expandable_segments:True"

#### Experiment Meta-Parameters ####
CHECKPOINT_POLICY=true
CHECKPOING_WM=false

TOTAL_TIMESTEPS=60000000
CHECKPOINT_FREQ=64000
LEARNING_RATE=0.0003
BATCH_SIZE_MOD=250
N_STEPS=1000
N_EPOCHS=5 
VALUE_COEFF=0.5

LP_HORIZON=0.9
MMM_HORIZON=0.99
BP_HORIZON=0.9
GRATING_SPEED=0.01

IDM_SCALE=1.0
TASK_SCALE=0.0
ACTION_PENALTY_SCALE=1.0
WORLD_MODEL_TYPE="mlp"
IDM_TYPE="3m_progress"
LOAD_DMC=false
USE_FLOW=false
BIAS_RESET=true

## must define world model path here
#wm_path=

PRETRAIN_ENV=drift_pi_wm
NAME="grating-${IDM_TYPE}-${WORLD_MODEL_TYPE}-idm-${IDM_SCALE}-task-${TASK_SCALE}-penalty-${ACTION_PENALTY_SCALE}-PRT_ENV-${PRETRAIN_ENV}-BIAS_RESET-${BIAS_RESET}"

##### Training Script #####

SCRATCH_DIR="/scratch/rdkeller"
SCRATCH_JOB_DIR="${SCRATCH_DIR}/${NAME}"
PERM_LOG_DIR=/data/user_data/rdkeller/rl_training_logs/${NAME}

echo "====== JOB ENVIRONMENT REPORT ======"
echo "Node: $SLURM_NODELIST"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "Scratch directory: ${SCRATCH_JOB_DIR}"
echo "Permanent directory: ${PERM_LOG_DIR}"
echo ""
echo "Disk space on /scratch:"
df -h /scratch
echo "======================================"

echo "Creating scratch directory: ${SCRATCH_JOB_DIR}"
mkdir -p ${SCRATCH_JOB_DIR}
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create scratch directory on node $SLURM_NODELIST"
    echo "This node has insufficient disk space. Consider excluding it in future jobs."
    echo "Disk space report:"
    df -h /scratch
    echo "Exiting job due to insufficient disk space."
    exit 1
fi

echo "Creating permanent log directory: ${PERM_LOG_DIR}"
mkdir -p ${PERM_LOG_DIR}
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create permanent log directory"
    echo "Exiting job."
    exit 1
fi

echo "Directories created successfully."
cleanup() {
    echo "Job is ending. Running cleanup operations..."
    echo "Copying current results from scratch to permanent storage..."
    rsync -av --ignore-existing ${SCRATCH_JOB_DIR}/ ${PERM_LOG_DIR}/
    echo "Copy complete. Files are stored in ${PERM_LOG_DIR}"
    echo "Removing scratch directory ${SCRATCH_JOB_DIR}..."
    rm -rf ${SCRATCH_JOB_DIR}
    echo "Cleanup complete."
}

trap cleanup EXIT INT TERM

python /home/rdkeller/zebrafish_agent/zfa_rl_agent/experiments/grating/run_experiment.py \
    name="$NAME" \
    total_timesteps="$TOTAL_TIMESTEPS" \
    checkpoint_save_freq="$CHECKPOINT_FREQ" \
    batch_mod="$BATCH_SIZE_MOD" \
    learning_rate="$LEARNING_RATE" \
    n_steps="$N_STEPS" \
    n_epochs="$N_EPOCHS" \
    vf_coef="$VALUE_COEFF" \
    ir_scale="$IDM_SCALE" \
    er_scale="$TASK_SCALE" \
    reward_type="$IDM_TYPE" \
    ap_scale="$ACTION_PENALTY_SCALE" \
    use_flow="$USE_FLOW" \
    world_model_class="$WORLD_MODEL_TYPE" \
    log_dir="${SCRATCH_JOB_DIR}" \
    job_id="${SLURM_JOB_ID}" \
    mmm_progress_horizon="$MMM_HORIZON" \
    learning_progress_horizon="$LP_HORIZON" \
    load_dmc_agent="$LOAD_DMC" \
    wm_path="$wm_path" \
    policy_path="$policy_path" \
    checkpointing="$CHECKPOINT_POLICY" \
    wm_checkpointing="$CHECKPOINT_WM" \
    cycle_horizon="$BP_HORIZON" \
    grating_speed="$GRATING_SPEED" \

echo "Training completed."