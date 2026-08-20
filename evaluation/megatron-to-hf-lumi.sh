#!/usr/bin/env bash
#SBATCH --job-name=megatron-to-hf
#SBATCH --account=project_465002530
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --time=00:15:00
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=../logs/convert/megatron-to-hf-%j.out
#SBATCH --error=../logs/convert/megatron-to-hf-%j.err

set -euo pipefail

# Run the local-path-aware converter with LUMI configuration.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# LUMI container
CONTAINER="/scratch/project_462001427/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif"

# Directories to bind
BIND_DIRS="/scratch,$(realpath /scratch/project_462001427)"

# Paths to Megatron-Bridge-LUMI and Megatron-Bridge-utils repos
BRIDGE_PATH="/scratch/project_462001427/tools/Megatron-Bridge-LUMI"
UTILS_PATH="/scratch/project_462001427/tools/Megatron-Bridge-utils"

if [[ $# -ne 4 ]]; then
    echo "Usage: $0 INPUT_PATH OUTPUT_PATH HF_MODEL TOKENIZER" >&2
    exit 1
fi

# If this script is run without sbatch, invoke with sbatch here.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    sbatch "$0" "$@"
    exit
fi

singularity exec \
    --bind "$BIND_DIRS" \
    "$CONTAINER" \
    "$SCRIPT_DIR/megatron-to-hf-local.sh" "$@" "$UTILS_PATH" "$BRIDGE_PATH"
