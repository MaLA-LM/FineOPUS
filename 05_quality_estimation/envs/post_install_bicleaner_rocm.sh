#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------------------------------
# Post-install: replace standard TensorFlow with ROCm 6.3-compatible build.
#
# bicleaner-ai hardcodes "tensorflow" as a dependency, which pulls in
# the CPU/CUDA version. This script force-removes it and installs
# the AMD official ROCm 6.3 wheel (matching LUMI's ROCm 6.3.4).
# -------------------------------------------------------------------

ROCM_INDEX="https://repo.radeon.com/rocm/manylinux/rocm-rel-6.3/"

echo "[post-install] Replacing standard TensorFlow with AMD ROCm 6.3 build..."

# 1. Force remove the standard TensorFlow packages pulled by bicleaner-ai
pip uninstall -y tensorflow tensorflow-intel tensorflow-cpu tensorflow-rocm 2>/dev/null || true

# 2. Install the ROCm-specific TensorFlow
# Note: Using --find-links because AMD's repo is a flat directory
pip install \
    --find-links "$ROCM_INDEX" \
    "tensorflow-rocm==2.17.0" \
    "numpy<2.0.0" \
    "protobuf==3.20.3"

echo "[post-install] tensorflow-rocm replacement complete."