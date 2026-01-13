#!/bin/bash
#SBATCH --job-name=jupyter_notebook
#SBATCH --partition=debug              # Choose the partition (cpu/gpu) based on your needs
#SBATCH --nodes=1                    # Number of nodes (usually 1 for Jupyter)
#SBATCH --ntasks=1                   # Number of tasks
#SBATCH --cpus-per-task=4           # Number of CPUs per task (adjust based on your requirements)
#SBATCH --mem=16GB                  # Memory per node (adjust as needed)
#SBATCH --time=02:00:00             # Max runtime (adjust as needed)

# Load any required modules (e.g., Python, Jupyter, etc.)
module load python/3.8   # Load Python (adjust the version based on your cluster)
module load jupyter      # Load Jupyter module if available

# Step 2: Activate the conda environment
source /home/akirscht/miniconda3/bin/activate zfa

# Start the Jupyter notebook server
jupyter-notebook --no-browser --port=8888 --ip=0.0.0.0
