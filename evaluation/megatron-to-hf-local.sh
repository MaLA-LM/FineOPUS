#!/usr/bin/env bash

# Export a Megatron torch_dist checkpoint to Hugging Face format.
# This is based on Megatron-Bridge-utils/megatron-to-hf.sh, with one important
# extension: TOKENIZER may be an arbitrary local directory.

set -euo pipefail

if (( $# < 4 || $# > 6 )); then
    echo "Usage: $0 INPUT_PATH OUTPUT_PATH HF_MODEL TOKENIZER [UTILS_PATH [BRIDGE_PATH]]" >&2
    exit 2
fi

INPUT_PATH="$1"
OUTPUT_PATH="$2"
HF_MODEL="$3"
HF_TOKENIZER="$4"
UTILS_PATH="${5:-/scratch/project_462001427/tools/Megatron-Bridge-utils}"
BRIDGE_PATH="${6:-/scratch/project_462001427/tools/Megatron-Bridge-LUMI}"

[[ -d "$INPUT_PATH" ]] || { echo "ERROR: checkpoint is not a directory: $INPUT_PATH" >&2; exit 1; }
[[ ! -e "$OUTPUT_PATH" ]] || { echo "ERROR: refusing to overwrite: $OUTPUT_PATH" >&2; exit 1; }

DUMMY_MODEL_SCRIPT="$UTILS_PATH/create_dummy_model.py"
PATCH_SCRIPT="$UTILS_PATH/export_custom_tokenizer_standalone.py"
CONVERT_SCRIPT="$BRIDGE_PATH/examples/conversion/convert_checkpoints.py"
CONFIG="$UTILS_PATH/configs/$HF_MODEL"
RUN_CONFIG="$UTILS_PATH/templates/$HF_MODEL/run_config.yaml"
TOKENIZER_IN="$UTILS_PATH/tokenizers/$HF_MODEL"
if [[ ! -d "$TOKENIZER_IN" && "$HF_MODEL" == openeurollm/Qwen3-*-ne ]]; then
    # The custom-size configs share this tokenizer for constructing the dummy
    # HF model. The actual training tokenizer is installed in step 4.
    TOKENIZER_IN="$UTILS_PATH/tokenizers/openeurollm/tokenizer-256k"
fi

if [[ -d "$HF_TOKENIZER" ]]; then
    TOKENIZER_OUT=$(realpath "$HF_TOKENIZER")
else
    TOKENIZER_OUT="$UTILS_PATH/tokenizers/$HF_TOKENIZER"
fi

for path in "$DUMMY_MODEL_SCRIPT" "$PATCH_SCRIPT" "$CONVERT_SCRIPT" "$CONFIG" "$RUN_CONFIG" "$TOKENIZER_IN" "$TOKENIZER_OUT"; do
    [[ -e "$path" ]] || { echo "ERROR: required conversion asset is missing: $path" >&2; exit 1; }
done

DUMMY_HF_MODEL_PATH=$(mktemp -d)
TMP_MEGATRON_ROOT=$(mktemp -d)
cleanup() {
    rm -rf -- "$DUMMY_HF_MODEL_PATH" "$TMP_MEGATRON_ROOT"
}
trap cleanup EXIT

echo "[convert 1/4] Creating dummy $HF_MODEL model"
python3 "$DUMMY_MODEL_SCRIPT" "$CONFIG" "$TOKENIZER_IN" "$DUMMY_HF_MODEL_PATH"

VOCAB_SIZE=$(python3 - "$TOKENIZER_OUT" <<'PY'
import sys
from transformers import AutoTokenizer
print(len(AutoTokenizer.from_pretrained(sys.argv[1], trust_remote_code=True)))
PY
)

echo "[convert 2/4] Staging checkpoint links; tokenizer vocab size: $VOCAB_SIZE"
# The checkpoint is read-only during export. A symlink tree lets us add the
# required run_config.yaml without copying all distcp shards to node-local /tmp.
cp -rs -- "$INPUT_PATH" "$TMP_MEGATRON_ROOT"
TMP_MEGATRON_PATH="$TMP_MEGATRON_ROOT/$(basename "$INPUT_PATH")"
perl -pe "s/<<<VOCAB_SIZE>>>/$VOCAB_SIZE/" "$RUN_CONFIG" > "$TMP_MEGATRON_PATH/run_config.yaml"

echo "[convert 3/4] Exporting weights to $OUTPUT_PATH"
export PYTHONPATH="$BRIDGE_PATH/python-packages:$BRIDGE_PATH/3rdparty/Megatron-LM:$BRIDGE_PATH/src:${PYTHONPATH:-}"
python "$CONVERT_SCRIPT" export \
    --megatron-path "$TMP_MEGATRON_PATH" \
    --hf-model "$DUMMY_HF_MODEL_PATH" \
    --hf-path "$OUTPUT_PATH"

echo "[convert 4/4] Installing the training tokenizer and patching config.json"
python "$PATCH_SCRIPT" --hf-path "$OUTPUT_PATH" --tokenizer-path "$TOKENIZER_OUT"

test -s "$OUTPUT_PATH/config.json"
find "$OUTPUT_PATH" -maxdepth 1 -type f -name '*.safetensors' -print -quit | grep -q .
echo "Conversion complete: $OUTPUT_PATH"
