#!/bin/bash
# Launch FLASK datagen for all models.
# Run from the judgment-to-noise directory on Jupiter.
#
# Pre-requisites:
#   1. Pre-download all models on the login node:
#      for m in meta-llama/Meta-Llama-3-8B meta-llama/Meta-Llama-3-8B-Instruct \
#               Magpie-Align/Llama-3-8B-Magpie-Align-SFT-v0.2 \
#               Magpie-Align/Llama-3-8B-Magpie-Align-v0.2 \
#               allenai/llama-3-tulu-2-dpo-8b \
#               jondurbin/bagel-8b-v1.0 facebook/opt-125m; do
#        python -c "from huggingface_hub import snapshot_download; snapshot_download('$m')"
#      done
#
#   2. Copy the input data:
#      cp /path/to/subsample_questions.jsonl data/flask/

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs outputs

MODELS=(
    "meta-llama/Meta-Llama-3-8B"
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "Magpie-Align/Llama-3-8B-Magpie-Align-SFT-v0.2"
    "Magpie-Align/Llama-3-8B-Magpie-Align-v0.2"
    "allenai/llama-3-tulu-2-dpo-8b"
    "jondurbin/bagel-8b-v1.0"
    "facebook/opt-125m"
)

# Note: Llama-3-8B-Tulu-330K and Llama-3-8B-WildChat are custom/unknown HF repos.
# Add them here once the exact repo IDs are confirmed:
#   "penfever/Llama-3-8B-Tulu-330K"
#   "penfever/Llama-3-8B-WildChat"

echo "Submitting ${#MODELS[@]} FLASK datagen jobs..."
echo ""

for MODEL in "${MODELS[@]}"; do
    SAFE=$(echo "$MODEL" | tr '/' '_' | tr '.' '-')
    JOB_ID=$(sbatch --job-name="flask_${SAFE}" run_flask_datagen.sbatch "$MODEL" | awk '{print $4}')
    echo "  Submitted $MODEL -> Job $JOB_ID"
done

echo ""
echo "Done. Monitor with: squeue -u \$USER | grep flask"
echo "Outputs will be in: outputs/"
