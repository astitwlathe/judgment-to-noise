#!/usr/bin/env bash
# Fire the 4 ajudge intervention jobs on GPT-4o-mini x MT-Bench.
#
# Prerequisites:
#   1. `conda activate ajudge`
#   2. `source /Users/benjaminfeuer/Documents/secrets.env`  (OPENAI_API_KEY)
#   3. `bash setup.sh`  (installs the 4 rubric JSONs into ajudge)
#
# Usage:  bash run_interventions.sh [--dry-run]
#
# Jobs write to /Users/benjaminfeuer/Documents/abb/data/mt_bench/jobs/
# Expected cost: <$5 total via OpenAI (1.7k judgments x 4 conditions).

set -e

DRY_RUN=""
if [[ "$1" == "--dry-run" ]]; then
  DRY_RUN="echo [DRY-RUN]"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIGS=(
  intervention_phrase
  intervention_voice
  intervention_lexicographic
  intervention_combined
)

# ajudge expects to be run from the abb/data parent so ./data/mt_bench/...
# paths in the YAML resolve correctly.
cd /Users/benjaminfeuer/Documents/abb

for cfg in "${CONFIGS[@]}"; do
  yaml_path="$SCRIPT_DIR/configs/$cfg.yaml"
  echo
  echo "=== Launching $cfg ==="
  $DRY_RUN ajudge job start "$yaml_path"
done

echo
echo "All 4 jobs submitted. Monitor with: ajudge job list"
echo "Results will land in: /Users/benjaminfeuer/Documents/abb/data/mt_bench/jobs/"
