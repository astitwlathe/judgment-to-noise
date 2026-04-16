#!/usr/bin/env bash
# Install the 4 intervention rubrics into the ajudge package so that
# `rubric_template: mt_bench_factored_<X>` resolves correctly.
#
# Re-run this after `pip install ajudge` (editable or otherwise) if the
# rubric set is ever reset.

set -e

# Locate ajudge's rubric dir via Python
AJUDGE_RUBRIC_DIR=$(conda run -n ajudge python -c \
  "import ajudge, os; print(os.path.join(os.path.dirname(ajudge.__file__), 'rubrics'))")

if [[ ! -d "$AJUDGE_RUBRIC_DIR" ]]; then
  echo "ERROR: ajudge rubric dir not found: $AJUDGE_RUBRIC_DIR" >&2
  exit 1
fi

echo "Installing rubrics into: $AJUDGE_RUBRIC_DIR"

SRC_DIR="$(cd "$(dirname "$0")" && pwd)/rubrics"

for f in "$SRC_DIR"/mt_bench_factored_*.json; do
  dst="$AJUDGE_RUBRIC_DIR/$(basename "$f")"
  if [[ -f "$dst" ]]; then
    echo "  overwriting existing: $(basename "$f")"
  else
    echo "  installing: $(basename "$f")"
  fi
  cp "$f" "$dst"
done

echo "Done. Installed rubrics:"
ls -1 "$AJUDGE_RUBRIC_DIR" | grep -E '^mt_bench_factored_(phrase_fixation|voice_bias|lexicographic|combined)\.json$' | sed 's/^/  /'
