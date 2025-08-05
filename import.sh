#!/bin/bash

REMOTE_BASE="md256895@orcuslogin1:/home/catC/md256895/orcus/PINNs"
LOCAL_BASE="/home/catC/md256895/PINNs"

FOLDERS=(
    "results"
)

EXCLUDES=(
    "--exclude=.git"
    "--exclude=__pycache__/"
    "--exclude=*.log"
)

for folder in "${FOLDERS[@]}"; do
    echo "Downloading $folder..."
    rsync -avzP "${EXCLUDES[@]}" "$REMOTE_BASE/$folder" "$LOCAL_BASE"
done
