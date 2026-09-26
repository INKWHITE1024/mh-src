#!/usr/bin/env bash
# Drive the full-loop pipeline from reviewer routes through collection, merge, and freeze.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/rethink-mh}"
PYTHON="${PYTHON:-python}"
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:?set the frozen initial-judgment run root}"
LITERACY_RUN_ROOT="${LITERACY_RUN_ROOT:?set five-fold Evidence Literacy run root}"
REVIEWER_ROUTES="${REVIEWER_ROUTES:-${LITERACY_RUN_ROOT}/outputs/evidence_literacy_oof/reviewer_routes.json}"
LITERACY_COMPLETE="${LITERACY_COMPLETE:-${LITERACY_RUN_ROOT}/outputs/evidence_literacy_oof/OOF_COMPLETE}"
PREPARED_ROOT="${PREPARED_ROOT:-${PROJECT_ROOT}/artifacts/full_loop/prepared}"
OUTPUT_GROUP="${OUTPUT_GROUP:-full_loop_raw_oof}"
MERGED_DIR="${MERGED_DIR:-${PROJECT_ROOT}/outputs/${OUTPUT_GROUP}/daic_woz/raw_oof_frozen}"
MANIFEST_PATH="${MANIFEST_PATH:-${PROJECT_ROOT}/outputs/full_loop_manifests/raw_oof.manifest.json}"
PREPARED_MANIFEST_PATH="${PREPARED_MANIFEST_PATH:-${PROJECT_ROOT}/outputs/full_loop_manifests/prepared.manifest.json}"
POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS:-20}"
STABLE_POLLS="${STABLE_POLLS:-3}"
MIN_FREE_MB="${MIN_FREE_MB:-40000}"
MAX_GPU_UTIL="${MAX_GPU_UTIL:-5}"
GPU_CANDIDATES="${GPU_CANDIDATES:-}"

if [[ ! "${POLL_INTERVAL_SECONDS}" =~ ^[0-9]+$ ]] \
  || (( POLL_INTERVAL_SECONDS <= 0 )); then
  echo "POLL_INTERVAL_SECONDS must be positive" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

wait_polls=0
while [[ ! -f "${LITERACY_COMPLETE}" || ! -f "${REVIEWER_ROUTES}" ]]; do
  wait_polls=$((wait_polls + 1))
  if (( wait_polls == 1 || wait_polls % 10 == 0 )); then
    echo "$(date -Is) waiting for five-fold Evidence Literacy reviewer routes."
  fi
  sleep "${POLL_INTERVAL_SECONDS}"
done
echo "$(date -Is) Evidence Literacy routes are complete; preparing full-loop views."

env \
  PROJECT_ROOT="${PROJECT_ROOT}" \
  PYTHON="${PYTHON}" \
  SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT}" \
  REVIEWER_ROUTES="${REVIEWER_ROUTES}" \
  OUTPUT_ROOT="${PREPARED_ROOT}" \
  "${PROJECT_ROOT}/scripts/prepare_full_loop.sh"

mkdir -p "$(dirname "${PREPARED_MANIFEST_PATH}")"
if [[ -f "${PREPARED_MANIFEST_PATH}" ]]; then
  "${PYTHON}" -m rethink_mh.experiments.frozen_artifact_manifest \
    --root "${PREPARED_ROOT}" \
    --verify-manifest "${PREPARED_MANIFEST_PATH}" >/dev/null
else
  "${PYTHON}" -m rethink_mh.experiments.frozen_artifact_manifest \
    --root "${PREPARED_ROOT}" \
    --output "${PREPARED_MANIFEST_PATH}" >/dev/null
fi

pids=()
labels=()
for fold in 0 1 2 3 4; do
  complete="${PROJECT_ROOT}/outputs/${OUTPUT_GROUP}/daic_woz/fold${fold}/PIPELINE_COMPLETE"
  if [[ -f "${complete}" ]]; then
    echo "$(date -Is) verified existing full-loop fold ${fold}; skipping."
    continue
  fi
  echo "$(date -Is) queueing full-loop raw collection fold ${fold}."
  env \
    PROJECT_ROOT="${PROJECT_ROOT}" \
    PYTHON="${PYTHON}" \
    PREPARED_INFERENCE_DIR="${PREPARED_ROOT}/inference" \
    OUTPUT_GROUP="${OUTPUT_GROUP}" \
    POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS}" \
    STABLE_POLLS="${STABLE_POLLS}" \
    MIN_FREE_MB="${MIN_FREE_MB}" \
    MAX_GPU_UTIL="${MAX_GPU_UTIL}" \
    GPU_CANDIDATES="${GPU_CANDIDATES}" \
    "${PROJECT_ROOT}/scripts/wait_for_gpu_and_train.sh" \
    daic_woz full-loop "${fold}" &
  pids+=("$!")
  labels+=("fold=${fold}")
done

failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[${index}]}"; then
    echo "full-loop waiter failed: ${labels[${index}]}" >&2
    failed=1
  fi
done
if (( failed != 0 )); then
  exit 5
fi

fold_args=()
for fold in 0 1 2 3 4; do
  fold_dir="${PROJECT_ROOT}/outputs/${OUTPUT_GROUP}/daic_woz/fold${fold}"
  if [[ ! -f "${fold_dir}/PIPELINE_COMPLETE" ]]; then
    echo "completed marker is missing for full-loop fold ${fold}" >&2
    exit 5
  fi
  fold_args+=(--fold-dir "${fold_dir}")
done

if [[ ! -e "${MERGED_DIR}" ]]; then
  "${PYTHON}" -m rethink_mh.experiments.loop_merge \
    --prepared-inference-dir "${PREPARED_ROOT}/inference" \
    "${fold_args[@]}" \
    --output-dir "${MERGED_DIR}" \
    --expected-fold-count 5 \
    --expected-session-count 107
fi

mkdir -p "$(dirname "${MANIFEST_PATH}")"
if [[ -f "${MANIFEST_PATH}" ]]; then
  "${PYTHON}" -m rethink_mh.experiments.frozen_artifact_manifest \
    --root "${MERGED_DIR}" \
    --verify-manifest "${MANIFEST_PATH}" >/dev/null
else
  "${PYTHON}" -m rethink_mh.experiments.frozen_artifact_manifest \
    --root "${MERGED_DIR}" \
    --output "${MANIFEST_PATH}" >/dev/null
fi

touch "${PROJECT_ROOT}/outputs/${OUTPUT_GROUP}/COLLECTION_COMPLETE"
echo "$(date -Is) five-fold label-free raw loop trajectories are frozen."
