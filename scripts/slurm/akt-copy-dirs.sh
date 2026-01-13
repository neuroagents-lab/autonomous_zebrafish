#!/bin/bash
#SBATCH --job-name=copy_dataset_50
#SBATCH --output=/data/user_data/akirscht/slurm/out/reload_files_%A_%a.out
#SBATCH --error=/data/user_data/akirscht/slurm/err/reload_files%A_%a.error
#SBATCH --partition=general
#SBATCH --nodelist=babel-7-1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=2-00:00:00

# Activate the Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate zfa

# Set dataset paths
SRC_DIR="/data/group_data/neuroagents_lab/training_datasets/zfa_pretraining_data/processed/MarineVideoKit"
SRC_FLOW_DIR="/data/group_data/neuroagents_lab/training_datasets/zfa_pretraining_data/processed/MarineVideoKit/processed/MarineVideoKit_flows"

DEST_DIR="/scratch/$USER/marine_video_kit_50"
DEST_FLOW_DIR="/scratch/$USER/marine_video_kit_50/processed/MarineVideoKit_flows"

NPZ_FILE="/data/user_data/akirscht/zfa/dataset_train_0.5.npz"

# Delete everything inside $USER
rm -rf /scratch/$USER/*

# Ensure output directories exist
mkdir -p "$DEST_DIR/train" "$DEST_DIR/val" "$DEST_DIR/test"
mkdir -p "$DEST_FLOW_DIR/train" "$DEST_FLOW_DIR/val" "$DEST_FLOW_DIR/test"

# Check available storage
AVAILABLE_SPACE=$(df --output=avail -BG "/scratch" | tail -1 | awk '{print $1}' | sed 's/G//')
REQUIRED_SPACE=3500  # Adjust based on dataset size estimation

if [[ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE" ]]; then
    echo "Error: Not enough storage in /scratch. Required: ${REQUIRED_SPACE}G, Available: ${AVAILABLE_SPACE}G."
    exit 1
fi

# Set a fixed random seed for reproducibility
SEED=42
export PYTHONHASHSEED=$SEED

# Run the Python script to select and copy 50% of the training dataset
python - <<EOF
import os
import numpy as np
import random
import shutil
from tqdm import tqdm

# Paths
SRC_DIR = "$SRC_DIR"
SRC_FLOW_DIR = "$SRC_FLOW_DIR"
DEST_DIR = "$DEST_DIR"
DEST_FLOW_DIR = "$DEST_FLOW_DIR"
SEED = $SEED

# Set seed for reproducibility
random.seed(SEED)

# Function to get subdirectories (locations)
def get_subdirectories(path):
    return [loc for loc in os.listdir(path) if os.path.isdir(os.path.join(path, loc))]

# Get full lists of train, val, and test locations
train_locations = get_subdirectories(os.path.join(SRC_DIR, "train"))
val_locations = get_subdirectories(os.path.join(SRC_DIR, "val"))
test_locations = get_subdirectories(os.path.join(SRC_DIR, "test"))

# Selected 50% of training locations
npz_data = np.load("$NPZ_FILE")
selected_train_locations = npz_data["sampled_dirs"]

# Function to copy data and flow directories with error handling
def copy_subset(split_name, selected_locs=None):
    src_video_dir = os.path.join(SRC_DIR, split_name)
    src_flow_dir = os.path.join(SRC_FLOW_DIR, split_name)
    dest_video_dir = os.path.join(DEST_DIR, split_name)
    dest_flow_dir = os.path.join(DEST_FLOW_DIR, split_name)

    os.makedirs(dest_video_dir, exist_ok=True)
    os.makedirs(dest_flow_dir, exist_ok=True)

    if selected_locs is None:
        selected_locs = get_subdirectories(src_video_dir)

    for loc in tqdm(selected_locs, desc=f"Copying {split_name}", unit="dir"):
        src_video_path = os.path.join(src_video_dir, loc)
        dest_video_path = os.path.join(dest_video_dir, loc)
        try:
            shutil.copytree(src_video_path, dest_video_path, dirs_exist_ok=True)
        except Exception as e:
            print(f"Error copying {src_video_path} to {dest_video_path}: {e}")

        # Copy flow data if it exists
        src_flow_path = os.path.join(src_flow_dir, loc)
        dest_flow_path = os.path.join(dest_flow_dir, loc)
        if os.path.exists(src_flow_path):
            try:
                shutil.copytree(src_flow_path, dest_flow_path, dirs_exist_ok=True)
            except Exception as e:
                print(f"Error copying {src_flow_path} to {dest_flow_path}: {e}")

# Copy 50% of train set
copy_subset("train", selected_train_locations)

# Copy all of val and test sets
copy_subset("val")
copy_subset("test")

EOF

if [[ $? -eq 0 ]]; then
    echo "Dataset copy completed successfully."
else
    echo "Dataset copy failed due to insufficient storage or another error."
fi

"""
#!/bin/sh
#SBATCH --job-name=copy_dataset_50
#SBATCH --output=/data/user_data/akirscht/slurm/out/copy_dirs%A_%a.out
#SBATCH --error=/data/user_data/akirscht/slurm/err/copy_dirs%A_%a.error
#SBATCH --partition=general
#SBATCH --nodelist=babel-7-1,babel-7-5,babel-4-13,babel-6-13
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=2-00:00:00

# Activate the Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate zfa

# Set random seed for reproducibility
RANDOM=42
INPUT_DIR="/data/group_data/neuroagents_lab/training_datasets/zfa_pretraining_data/processed/MarineVideoKit/train/"
OUTPUT_DIR="/data/user_data/akirscht/zfa"

# Function to extract name from directory
extract_name() {
    echo "$1" | rev | cut -d'_' -f2- | rev
}

# Create associative array to group directories by name
declare -A grouped_dirs

# Iterate through directories in the INPUT_DIR and group them
for dir in "$INPUT_DIR"*/; do
    name=$(extract_name "${dir%/}")
    grouped_dirs["$name"]+="$dir "
done

# Sample half of the directories for each name
sampled_dirs=()
for name in "${!grouped_dirs[@]}"; do
    dirs=(${grouped_dirs["$name"]})
    count=$((${#dirs[@]} / 2))
    if [ $count -eq 0 ]; then
        count=1
    fi
    sampled=$(printf "%s\n" "${dirs[@]}" | shuf -n $count)
    sampled_dirs+=($sampled)
done

# Create a Python script to save the sampled directories to a .npz file
python3 << EOF
import numpy as np
import os

sampled_dirs = """${sampled_dirs[@]}"""
sampled_dirs = sampled_dirs.split()

# Remove the INPUT_DIR prefix from the sampled directories
sampled_dirs = [os.path.relpath(dir, '$INPUT_DIR') for dir in sampled_dirs]

np.savez('$OUTPUT_DIR/dataset_train_0.5.npz', sampled_dirs=sampled_dirs)
print("Sampled directories saved to $OUTPUT_DIR/dataset_train_0.5.npz")
EOF
"""


