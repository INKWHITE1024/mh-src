#!/usr/bin/env bash
# Prepare physically separated label-free inference and outcome views for full-loop OOF.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/rethink-mh}"
PYTHON="${PYTHON:-python}"
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:?set the frozen initial-judgment run root}"
REVIEWER_ROUTES="${REVIEWER_ROUTES:?set complete five-fold reviewer routes}"
COMPARISON_ROOT="${COMPARISON_ROOT:-${SOURCE_RUN_ROOT}/outputs/matched_oof_comparison/daic_woz}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/artifacts/full_loop/prepared}"

required=(
  "${COMPARISON_ROOT}/paired_predictions.jsonl"
  "${COMPARISON_ROOT}/summary.json"
  "${REVIEWER_ROUTES}"
  "${SOURCE_RUN_ROOT}/artifacts/native_from_scratch"
)
for path in "${required[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "full-loop preparation input is missing: ${path}" >&2
    exit 3
  fi
done

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -e "${OUTPUT_ROOT}" ]]; then
  "${PYTHON}" - "${OUTPUT_ROOT}" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
if manifest.get("preparation_protocol_version") != "loop-prepare":
    raise SystemExit("existing full-loop preparation has another protocol")
for relative, record in manifest.get("files", {}).items():
    path = root / relative
    if not path.is_file():
        raise SystemExit(f"prepared file is missing: {path}")
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != record.get("sha256"):
        raise SystemExit(f"prepared file changed: {path}")
inference = root / "inference"
if {path.name for path in inference.iterdir()} != {
    "inference_plan.jsonl",
    "manifest.json",
}:
    raise SystemExit("inference capability directory contains extra files")
PY
  echo "$(date -Is) verified existing full-loop separated views."
  exit 0
fi

"${PYTHON}" -m rethink_mh.experiments.loop_prepare \
  --dataset daic_woz \
  --paired-predictions "${COMPARISON_ROOT}/paired_predictions.jsonl" \
  --comparison-summary "${COMPARISON_ROOT}/summary.json" \
  --reviewer-routes "${REVIEWER_ROUTES}" \
  --source-run-root "${SOURCE_RUN_ROOT}" \
  --output-dir "${OUTPUT_ROOT}" \
  --seed 42 \
  --expected-fold-count 5 \
  --expected-session-count 107

echo "$(date -Is) full-loop inference/outcome capability views prepared."
