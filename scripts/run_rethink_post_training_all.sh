#!/usr/bin/env bash
# Run the offline post-training pipeline for all datasets and emit a combined report.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/rethink-mh}"
GPU="${GPU:?set GPU to one physical GPU index}"
PYTHON="${PYTHON:-python}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-rethink_revision_post_training_v2_balanced}"
PACKAGE_ROOT="${PACKAGE_ROOT:-${PROJECT_ROOT}/outputs/rethink_post_training_v1}"
ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/rethink_post_training_v2_balanced}"
LOCK="${ROOT}/post_training.lock"
mkdir -p "$(dirname "${LOCK}")"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "another rethink post-training pipeline holds ${LOCK}" >&2
  exit 9
fi

failures=()
for dataset in daic_woz e_daic d_vlog; do
  if ! GPU="${GPU}" PROJECT_ROOT="${PROJECT_ROOT}" \
    PACKAGE_ROOT="${PACKAGE_ROOT}" OUTPUT_ROOT="${ROOT}" \
    EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
    "${PROJECT_ROOT}/scripts/run_rethink_post_training_pipeline.sh" "${dataset}"; then
    failures+=("${dataset}")
    echo "$(date -Is) ${dataset} failed or was gated; continuing independent datasets" >&2
  fi
done

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
report_status=0
"${PYTHON}" -m rethink_mh.experiments.rethink_post_training_report \
  --root "${ROOT}" \
  --package-root "${PACKAGE_ROOT}" \
  --experiment-name "${EXPERIMENT_NAME}" \
  --output-json "${ROOT}/results/summary.json" \
  --output-md "${ROOT}/results/summary.md" || report_status=$?

if (( ${#failures[@]} > 0 || report_status != 0 )); then
  "${PYTHON}" - "${ROOT}/PIPELINE_FAILURES.json" "${failures[@]}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

output, *datasets = sys.argv[1:]
Path(output).write_text(
    json.dumps(
        {
            "failed_or_gated_datasets": datasets,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        },
        indent=2,
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)
PY
  exit 6
fi
touch "${ROOT}/ALL_PIPELINES_COMPLETE"
