#!/bin/bash
# Script to download TAO dataset frames

echo "Downloading TAO dataset frames from HuggingFace..."
echo "This is a large dataset and may take several hours."
echo ""

# Activate conda environment
source ~/miniconda/etc/profile.d/conda.sh
conda activate masaenv

# Set download directory
TAO_ROOT="/home/dima/projects/TRACT/data/tao"
cd "$TAO_ROOT"

# Note: You may need to log in to HuggingFace first if not already done
# huggingface-cli login

# Download using huggingface-cli
echo "Starting download..."
huggingface-cli download \
    chengyenhsieh/TAO-Amodal \
    --repo-type dataset \
    --local-dir "$TAO_ROOT" \
    --local-dir-use-symlinks False

echo ""
echo "Download complete!"
echo "TAO dataset is now in: $TAO_ROOT"
