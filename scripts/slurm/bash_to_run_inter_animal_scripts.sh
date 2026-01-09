#!/bin/bash

# Directory containing your SLURM scripts
SCRIPTS_DIR="/om2/user/leokoz8/code/virtual_zebrafish/bash_scripts/slurm/inter_animal_slurm_scripts"

# Loop through all the SLURM scripts in the directory
for script in "$SCRIPTS_DIR"/*.sh; do
    if [[ -f $script ]]; then
        echo "Submitting $script..."
        sbatch "$script"
        
        # Optionally: Sleep for a few seconds before submitting the next job
        sleep 2s
    fi
done