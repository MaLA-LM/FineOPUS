#!/usr/bin/env bash
set -euo pipefail

# Keep core HF stack compatible with transformers<4.54
python -m pip install --upgrade \
  "huggingface_hub>=0.30,<1.0" \
  "safetensors>=0.4.3"

# Fail early if any dependency set is inconsistent
python -m pip check
