#!/usr/bin/env bash
# Collect one fold's label-free full-loop trajectories on a chosen GPU.
set -euo pipefail

GPU="${GPU:?set one physical GPU index}"
FOLD_INDEX="${FOLD_INDEX:?set outer FOLD_INDEX}"
PROJECT_ROOT="${PROJECT_ROOT:-/path/to/rethink-mh}"
PYTHON="${PYTHON:-python}"
PREPARED_INFERENCE_DIR="${PREPARED_INFERENCE_DIR:-${PROJECT_ROOT}/artifacts/full_loop/prepared/inference}"
OUTPUT_GROUP="${OUTPUT_GROUP:-full_loop_raw_oof}"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/${OUTPUT_GROUP}/daic_woz/fold${FOLD_INDEX}"
TASK_PATH="${TASK_PATH:-${PROJECT_ROOT}/configs/tasks/phq8_depression_binary.json}"
LOG_ROOT="${PROJECT_ROOT}/outputs/logs"

if [[ ! "${FOLD_INDEX}" =~ ^[0-9]+$ ]] || (( FOLD_INDEX < 0 || FOLD_INDEX > 4 )); then
  echo "FOLD_INDEX must be an integer from 0 to 4" >&2
  exit 2
fi
for required in \
  "${PREPARED_INFERENCE_DIR}/manifest.json" \
  "${PREPARED_INFERENCE_DIR}/inference_plan.jsonl" \
  "${TASK_PATH}"; do
  if [[ ! -f "${required}" ]]; then
    echo "full-loop collection input is missing: ${required}" >&2
    exit 3
  fi
done
if [[ -d "${OUTPUT_DIR}" && ! -f "${OUTPUT_DIR}/manifest.json" ]]; then
  echo "partial full-loop fold requires manual audit: ${OUTPUT_DIR}" >&2
  exit 4
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${LOG_ROOT}"

if [[ ! -f "${OUTPUT_DIR}/manifest.json" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
    -m rethink_mh.experiments.loop_collect \
    --prepared-dir "${PREPARED_INFERENCE_DIR}" \
    --task "${TASK_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --fold-index "${FOLD_INDEX}" \
    --device cuda:0 \
    --dtype bfloat16 \
    --attention-implementation sdpa \
    --max-new-tokens 640 \
    --max-input-tokens 8192 \
    --max-contract-retries 2 \
    2>&1 | tee \
      "${LOG_ROOT}/full_loop.fold${FOLD_INDEX}.collect.log"
fi

"${PYTHON}" - "${OUTPUT_DIR}" "${FOLD_INDEX}" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
fold = int(sys.argv[2])
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
raw = root / "trajectories.raw.jsonl"
if any(
    (
        manifest.get("collection_protocol_version")
        != "loop-collection",
        manifest.get("outer_fold") != fold,
        manifest.get("limited_check") is not False,
        not raw.is_file(),
        hashlib.sha256(raw.read_bytes()).hexdigest()
        != manifest.get("files", {})
        .get("trajectories.raw.jsonl", {})
        .get("sha256"),
    )
):
    raise SystemExit(f"invalid completed full-loop fold: {root}")
PY

touch "${OUTPUT_DIR}/PIPELINE_COMPLETE"
echo "$(date -Is) full-loop raw collection complete: fold=${FOLD_INDEX}."
