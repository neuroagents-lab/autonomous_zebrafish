#!/bin/bash

# Append "/raw" to the target directory to save into the target directory
pretraining_dataset_dir=$(python -c "from zfa.core.default_dirs import PRETRAINING_DATASET_DIR; print(PRETRAINING_DATASET_DIR)")

# Ensure the "raw" directory exists under the target directory
mkdir -p "$pretraining_dataset_dir/raw"

# Set the target directory to include "raw"
target_directory="$pretraining_dataset_dir/raw"

# Set the base URL of the website you want to download
base_url="https://hkust-vgd.ust.hk/MarineVideoKit/"

# Set your username and password for HTTP basic authentication
username="mvk"
password="547193"

# Use wget to recursively download the website's contents
wget --recursive --no-clobber --page-requisites --html-extension \
     --convert-links --restrict-file-names=windows --domains hkust-vgd.ust.hk \
     --no-parent --auth-no-challenge --http-user=$username --http-password=$password \
     $base_url -P "$target_directory" -o "$target_directory/wget.log"

wget_status=$?  # Save the exit status of wget

# Check the status
if [ $wget_status -eq 0 ]; then
    echo "wget completed successfully."
    # Proceed with the rest of your script
else
    echo "wget encountered an error with exit status: $wget_status"
    # Handle the error, exit, or take other appropriate action
fi
