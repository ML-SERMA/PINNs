#!/bin/bash

# Define local base and remote base
LOCAL_BASE="/home/catC/md256895/PINNs"
REMOTE_BASE="/home/catC/md256895/orcus/PINNs"
REMOTE_HOST="md256895@orcuslogin1"

# List of folders to sync
FOLDERS=(
    "data"
    "datagenerators"
    "experiments" 
    "benchmarks" 
    "pyPINNs"
)

# Exclude rules
EXCLUDES=(
    "--exclude=.git"
    "--exclude=__pycache__/"
    "--exclude=*.log"
    "--exclude=*.pt"
)

# Sync loop
for folder in "${FOLDERS[@]}"; do
    echo "Syncing $folder..."

    # Ensure remote folder exists
    # ssh "$REMOTE_HOST" "mkdir -p \"$REMOTE_BASE/$folder\""
    ssh "$REMOTE_HOST" "mkdir -p \"$REMOTE_BASE\""

    # Run rsync
    rsync -avzP "${EXCLUDES[@]}" "$LOCAL_BASE/$folder/" "$REMOTE_HOST:$REMOTE_BASE/$folder/"
done

# Also sync top-level *.sh files in LOCAL_BASE
echo "Syncing top-level *.sh files..."
rsync -avzP "${EXCLUDES[@]}" "$LOCAL_BASE/"*.sh "$REMOTE_HOST:$REMOTE_BASE/"
