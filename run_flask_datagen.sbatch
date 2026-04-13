#!/bin/bash
#SBATCH --time=11:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --output=logs/flask_datagen_%j.out
#SBATCH --job-name=flask_datagen
#SBATCH --account=reformo

# ============================================
# FLASK Datagen — Single-Node 8B Model
# ============================================
# Starts a vLLM server on one node, then generates
# completions for FLASK subsample questions.
#
# Usage:
#   sbatch run_flask_datagen.sbatch <model_hf_id>
#
# Example:
#   sbatch run_flask_datagen.sbatch meta-llama/Meta-Llama-3-8B-Instruct
# ============================================

MODEL="${1:?Usage: sbatch run_flask_datagen.sbatch <model_hf_id>}"

# Derive safe name for output files
MODEL_SAFE=$(echo "$MODEL" | tr '/' '_' | tr '.' '-')

echo "============================================"
echo "FLASK Datagen — Single Node"
echo "============================================"
echo "Model:   $MODEL"
echo "Safe:    $MODEL_SAFE"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "============================================"

# -----------------------------------------------
# Step 1: Environment setup
# -----------------------------------------------
source /e/scratch/jureap59/feuer1/miniforge3/etc/profile.d/conda.sh
conda activate otagent

if [ -f ~/secrets.env ]; then
    source ~/secrets.env
fi

REPO_DIR="${DCFT:-/e/scratch/jureap59/feuer1/OpenThoughts-Agent}"
if [ -f "$REPO_DIR/hpc/dotenv/jupiter.env" ]; then
    source "$REPO_DIR/hpc/dotenv/jupiter.env"
fi

export HF_HUB_CACHE="${HF_HUB_CACHE:-/e/data1/datasets/playground/ot/hf_hub}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Load CUDA module
module purge 2>/dev/null || true
module load nvidia-compilers/25.9-CUDA-13 2>&1 || true

WORK_DIR="/e/scratch/jureap59/feuer1/judgment-to-noise"
cd "$WORK_DIR" || { echo "ERROR: Cannot cd to $WORK_DIR"; exit 1; }
mkdir -p logs outputs

echo "HF_HUB_CACHE: $HF_HUB_CACHE"
echo "Python: $(python3 --version 2>&1)"
echo "Work dir: $WORK_DIR"

set -eo pipefail

# -----------------------------------------------
# Step 2: Start vLLM server (TP=4 for 8B on GH200)
# -----------------------------------------------
echo ""
echo "=== Step 2: Starting vLLM Server ==="

VLLM_PORT=8000
VLLM_API_KEY="token-flask-${SLURM_JOB_ID}"
VLLM_LOG="logs/vllm_${MODEL_SAFE}_${SLURM_JOB_ID}.log"

# 8B models fit easily on 4x GH200 with TP=4
# Use TP=1 for tiny models (opt-125m), TP=4 for 8B
PARAMS=$(python3 -c "print(int('$MODEL'.find('opt-125m') >= 0))")
if [ "$PARAMS" -eq 1 ]; then
    TP=1
    MAX_MODEL_LEN=2048
    MAX_TOKENS=512
else
    TP=4
    MAX_MODEL_LEN=4096
    MAX_TOKENS=2048
fi

# Detect base (non-chat) models that need /completions API
IS_BASE=$(python3 -c "
m = '$MODEL'
chat_markers = ['Instruct', 'Magpie-Align', 'tulu', 'bagel', 'chat', 'Chat']
print(int(not any(k in m for k in chat_markers)))
")
COMPLETIONS_FLAG=""
if [ "$IS_BASE" -eq 1 ]; then
    COMPLETIONS_FLAG="--completions-api"
    echo "Detected base model -> using /completions API"
fi

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port $VLLM_PORT \
    --api-key "$VLLM_API_KEY" \
    --tensor-parallel-size $TP \
    --gpu-memory-utilization 0.90 \
    --max-model-len $MAX_MODEL_LEN \
    --max-num-seqs 64 \
    --enable-prefix-caching \
    --trust-remote-code \
    >> "$VLLM_LOG" 2>&1 &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID (TP=$TP)"
echo "vLLM log: $VLLM_LOG"

# Wait for vLLM to become healthy
echo "Waiting for vLLM server to start..."
for i in {1..60}; do
    if curl -s "http://localhost:${VLLM_PORT}/v1/models" \
         -H "Authorization: Bearer ${VLLM_API_KEY}" > /dev/null 2>&1; then
        echo "vLLM server is ready! (took ~$((i * 5))s)"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM process died. Check $VLLM_LOG"
        tail -30 "$VLLM_LOG"
        exit 1
    fi
    if [ "$i" -eq 60 ]; then
        echo "ERROR: vLLM server failed to start after 5 minutes"
        tail -30 "$VLLM_LOG"
        kill $VLLM_PID 2>/dev/null || true
        exit 1
    fi
    sleep 5
done

# -----------------------------------------------
# Step 3: Generate completions
# -----------------------------------------------
echo ""
echo "=== Step 3: Generating Completions ==="

INPUT_FILE="data/flask/subsample_questions.jsonl"
OUTPUT_FILE="outputs/${MODEL_SAFE}_completions.jsonl"

python3 generate_completions.py \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FILE" \
    --model "$MODEL" \
    --base-url "http://localhost:${VLLM_PORT}/v1" \
    --api-key "$VLLM_API_KEY" \
    --max-tokens $MAX_TOKENS \
    --temperature 0.7 \
    $COMPLETIONS_FLAG

GEN_EXIT=$?

# -----------------------------------------------
# Step 4: Cleanup
# -----------------------------------------------
echo ""
echo "=== Step 4: Cleanup ==="

kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

echo ""
echo "============================================"
if [ $GEN_EXIT -eq 0 ]; then
    echo "FLASK Datagen COMPLETED: $MODEL"
    echo "  Output: $OUTPUT_FILE"
else
    echo "FLASK Datagen FAILED: $MODEL (exit code: $GEN_EXIT)"
fi
echo "============================================"

exit $GEN_EXIT
