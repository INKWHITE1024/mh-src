#!/usr/bin/env bash
# Run the full-loop offline sequence: freeze, evaluate, build the training package, and post-train.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/rethink-mh}"
PYTHON="${PYTHON:-python}"
COLLECTION_GROUP="${COLLECTION_GROUP:-full_loop_raw_oof}"
COLLECTION_COMPLETE="${COLLECTION_COMPLETE:-${PROJECT_ROOT}/outputs/${COLLECTION_GROUP}/COLLECTION_COMPLETE}"
RAW_OOF_DIR="${RAW_OOF_DIR:-${PROJECT_ROOT}/outputs/${COLLECTION_GROUP}/daic_woz/raw_oof_frozen}"
RAW_MANIFEST="${RAW_MANIFEST:-${PROJECT_ROOT}/outputs/full_loop_manifests/raw_oof.manifest.json}"
PREPARED_ROOT="${PREPARED_ROOT:-${PROJECT_ROOT}/artifacts/full_loop/prepared}"
PREPARED_MANIFEST="${PREPARED_MANIFEST:-${PROJECT_ROOT}/outputs/full_loop_manifests/prepared.manifest.json}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/experiments/full_loop.yaml}"
TASK="${TASK:-${PROJECT_ROOT}/configs/tasks/phq8_depression_binary.json}"
EVALUATION_DIR="${EVALUATION_DIR:-${PROJECT_ROOT}/outputs/full_loop_offline/daic_woz}"
PACKAGE_DIR="${PACKAGE_DIR:-${PROJECT_ROOT}/outputs/full_loop_training_package/daic_woz}"
TRAINING_DIR="${TRAINING_DIR:-${PROJECT_ROOT}/outputs/full_loop_post_training/daic_woz}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${PROJECT_ROOT}/outputs/full_loop_manifests}"
POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS:-20}"
STABLE_POLLS="${STABLE_POLLS:-3}"
MIN_FREE_MB="${MIN_FREE_MB:-40000}"
MAX_GPU_UTIL="${MAX_GPU_UTIL:-5}"
GPU_CANDIDATES="${GPU_CANDIDATES:-}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

wait_polls=0
while [[ ! -f "${COLLECTION_COMPLETE}" ]]; do
  wait_polls=$((wait_polls + 1))
  if (( wait_polls == 1 || wait_polls % 10 == 0 )); then
    echo "$(date -Is) waiting for frozen raw full-loop OOF trajectories."
  fi
  sleep "${POLL_INTERVAL_SECONDS}"
done

for required in \
  "${RAW_OOF_DIR}/RAW_OOF_FROZEN" \
  "${RAW_MANIFEST}" \
  "${PREPARED_ROOT}/outcomes/outcomes.jsonl" \
  "${PREPARED_MANIFEST}" \
  "${CONFIG}" \
  "${TASK}"; do
  if [[ ! -f "${required}" ]]; then
    echo "post-collection input is missing: ${required}" >&2
    exit 3
  fi
done

if [[ -d "${EVALUATION_DIR}" && ! -f "${EVALUATION_DIR}/summary.json" ]]; then
  echo "partial offline evaluation requires manual audit: ${EVALUATION_DIR}" >&2
  exit 4
fi
if [[ ! -f "${EVALUATION_DIR}/summary.json" ]]; then
  "${PYTHON}" -m rethink_mh.experiments.loop_evaluate \
    --raw-oof-dir "${RAW_OOF_DIR}" \
    --raw-artifact-manifest "${RAW_MANIFEST}" \
    --prepared-root "${PREPARED_ROOT}" \
    --prepared-artifact-manifest "${PREPARED_MANIFEST}" \
    --config "${CONFIG}" \
    --output-dir "${EVALUATION_DIR}"
fi

mkdir -p "${MANIFEST_ROOT}"
EVALUATION_MANIFEST="${MANIFEST_ROOT}/offline_evaluation.manifest.json"
if [[ -f "${EVALUATION_MANIFEST}" ]]; then
  "${PYTHON}" -m rethink_mh.experiments.frozen_artifact_manifest \
    --root "${EVALUATION_DIR}" \
    --verify-manifest "${EVALUATION_MANIFEST}" >/dev/null
else
  "${PYTHON}" -m rethink_mh.experiments.frozen_artifact_manifest \
    --root "${EVALUATION_DIR}" \
    --output "${EVALUATION_MANIFEST}" >/dev/null
fi

if [[ -d "${PACKAGE_DIR}" && ! -f "${PACKAGE_DIR}/manifest.json" ]]; then
  echo "partial training package requires manual audit: ${PACKAGE_DIR}" >&2
  exit 4
fi
if [[ ! -f "${PACKAGE_DIR}/manifest.json" ]]; then
  "${PYTHON}" -m rethink_mh.experiments.loop_training_package \
    --raw-oof-dir "${RAW_OOF_DIR}" \
    --raw-artifact-manifest "${RAW_MANIFEST}" \
    --prepared-root "${PREPARED_ROOT}" \
    --prepared-artifact-manifest "${PREPARED_MANIFEST}" \
    --evaluation-dir "${EVALUATION_DIR}" \
    --config "${CONFIG}" \
    --task "${TASK}" \
    --output-dir "${PACKAGE_DIR}"
fi

PACKAGE_MANIFEST="${MANIFEST_ROOT}/training_package.manifest.json"
if [[ -f "${PACKAGE_MANIFEST}" ]]; then
  "${PYTHON}" -m rethink_mh.experiments.frozen_artifact_manifest \
    --root "${PACKAGE_DIR}" \
    --verify-manifest "${PACKAGE_MANIFEST}" >/dev/null
else
  "${PYTHON}" -m rethink_mh.experiments.frozen_artifact_manifest \
    --root "${PACKAGE_DIR}" \
    --output "${PACKAGE_MANIFEST}" >/dev/null
fi

env \
  PROJECT_ROOT="${PROJECT_ROOT}" \
  PYTHON="${PYTHON}" \
  PACKAGE_DIR="${PACKAGE_DIR}" \
  OUTPUT_DIR="${TRAINING_DIR}" \
  POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS}" \
  STABLE_POLLS="${STABLE_POLLS}" \
  MIN_FREE_MB="${MIN_FREE_MB}" \
  MAX_GPU_UTIL="${MAX_GPU_UTIL}" \
  GPU_CANDIDATES="${GPU_CANDIDATES}" \
  "${PROJECT_ROOT}/scripts/wait_for_gpu_and_train.sh" \
  daic_woz loop-train

TRAINING_MANIFEST="${MANIFEST_ROOT}/post_training.manifest.json"
if [[ -f "${TRAINING_MANIFEST}" ]]; then
  "${PYTHON}" -m rethink_mh.experiments.frozen_artifact_manifest \
    --root "${TRAINING_DIR}" \
    --verify-manifest "${TRAINING_MANIFEST}" >/dev/null
else
  "${PYTHON}" -m rethink_mh.experiments.frozen_artifact_manifest \
    --root "${TRAINING_DIR}" \
    --output "${TRAINING_MANIFEST}" >/dev/null
fi

touch "${PROJECT_ROOT}/outputs/full_loop_post_training/POST_SEQUENCE_COMPLETE"
echo "$(date -Is) outcome join and queued post-training sequence complete."
