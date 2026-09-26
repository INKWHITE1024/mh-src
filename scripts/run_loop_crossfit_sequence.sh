#!/usr/bin/env bash
# Drive cross-fitted revision-policy training and prediction across all folds.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/rethink-mh}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}}"
PYTHON="${PYTHON:-python}"
FULL_LOOP_RUN_ROOT="${FULL_LOOP_RUN_ROOT:?set the frozen full-loop run root}"
CROSS_ROOT="${CROSS_ROOT:-${RUN_ROOT}/outputs/loop_crossfit}"
RAW_OOF_DIR="${RAW_OOF_DIR:-${RUN_ROOT}/outputs/contract_recovery_raw_oof/daic_woz/raw_oof_frozen}"
RAW_ARTIFACT_MANIFEST="${RAW_ARTIFACT_MANIFEST:-${RUN_ROOT}/outputs/contract_recovery_manifests/raw_oof.manifest.json}"
EVALUATION_DIR="${EVALUATION_DIR:-${RUN_ROOT}/outputs/contract_recovery_offline/daic_woz}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/experiments/full_loop.yaml}"
TASK_PATH="${TASK_PATH:-${PROJECT_ROOT}/configs/tasks/phq8_depression_binary.json}"
PREPARED_ROOT="${PREPARED_ROOT:-${FULL_LOOP_RUN_ROOT}/artifacts/full_loop/prepared}"
PREPARED_INFERENCE_DIR="${PREPARED_ROOT}/inference"
PREPARED_ARTIFACT_MANIFEST="${PREPARED_ARTIFACT_MANIFEST:-${FULL_LOOP_RUN_ROOT}/outputs/full_loop_manifests/prepared.manifest.json}"
POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS:-20}"
STABLE_POLLS="${STABLE_POLLS:-3}"
MIN_FREE_MB="${MIN_FREE_MB:-30000}"
MAX_GPU_UTIL="${MAX_GPU_UTIL:-100}"
GPU_CANDIDATES="${GPU_CANDIDATES:-0}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p \
  "${CROSS_ROOT}/packages" \
  "${CROSS_ROOT}/manifests" \
  "${CROSS_ROOT}/logs" \
  "${RUN_ROOT}/control/gpu_locks"

for required in \
  "${RAW_OOF_DIR}/RAW_OOF_FROZEN" \
  "${RAW_ARTIFACT_MANIFEST}" \
  "${EVALUATION_DIR}/summary.json" \
  "${EVALUATION_DIR}/training_decisions.jsonl" \
  "${PREPARED_ROOT}/outcomes/outcomes.jsonl" \
  "${PREPARED_ARTIFACT_MANIFEST}" \
  "${CONFIG_PATH}" \
  "${TASK_PATH}"; do
  if [[ ! -f "${required}" ]]; then
    echo "crossfit sequence input is missing: ${required}" >&2
    exit 3
  fi
done

for fold in 0 1 2 3 4; do
  package="${CROSS_ROOT}/packages/fold${fold}"
  package_manifest="${CROSS_ROOT}/manifests/package.fold${fold}.manifest.json"
  if [[ -d "${package}" && ! -f "${package}/manifest.json" ]]; then
    echo "partial crossfit package requires manual audit: ${package}" >&2
    exit 4
  fi
  if [[ ! -f "${package}/manifest.json" ]]; then
    "${PYTHON}" \
      -m rethink_mh.experiments.loop_training_package \
      --raw-oof-dir "${RAW_OOF_DIR}" \
      --raw-artifact-manifest "${RAW_ARTIFACT_MANIFEST}" \
      --prepared-root "${PREPARED_ROOT}" \
      --prepared-artifact-manifest "${PREPARED_ARTIFACT_MANIFEST}" \
      --evaluation-dir "${EVALUATION_DIR}" \
      --config "${CONFIG_PATH}" \
      --task "${TASK_PATH}" \
      --output-dir "${package}" \
      --heldout-fold "${fold}" >/dev/null
  fi
  if [[ ! -f "${package_manifest}" ]]; then
    "${PYTHON}" \
      -m rethink_mh.experiments.frozen_artifact_manifest \
      --root "${package}" \
      --output "${package_manifest}" >/dev/null
  else
    "${PYTHON}" \
      -m rethink_mh.experiments.frozen_artifact_manifest \
      --root "${package}" \
      --verify-manifest "${package_manifest}" >/dev/null
  fi
done


pids=()
for fold in 0 1 2 3 4; do
  env \
    PROJECT_ROOT="${PROJECT_ROOT}" \
    RUN_ROOT="${RUN_ROOT}" \
    PYTHON="${PYTHON}" \
    CROSS_ROOT="${CROSS_ROOT}" \
    RAW_OOF_DIR="${RAW_OOF_DIR}" \
    RAW_ARTIFACT_MANIFEST="${RAW_ARTIFACT_MANIFEST}" \
    PREPARED_INFERENCE_DIR="${PREPARED_INFERENCE_DIR}" \
    TASK_PATH="${TASK_PATH}" \
    POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS}" \
    STABLE_POLLS="${STABLE_POLLS}" \
    MIN_FREE_MB="${MIN_FREE_MB}" \
    MAX_GPU_UTIL="${MAX_GPU_UTIL}" \
    GPU_CANDIDATES="${GPU_CANDIDATES}" \
    LOCK_ROOT="${RUN_ROOT}/control/gpu_locks" \
    "${PROJECT_ROOT}/scripts/wait_for_gpu_and_train.sh" \
    daic_woz loop-crossfit "${fold}" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if (( failed != 0 )); then
  echo "one or more crossfit fold jobs failed" >&2
  exit 5
fi

frozen="${CROSS_ROOT}/predictions_frozen"
fold_args=()
for fold in 0 1 2 3 4; do
  fold_args+=(--fold-dir "${CROSS_ROOT}/predictions/fold${fold}")
done
if [[ ! -e "${frozen}" ]]; then
  "${PYTHON}" -m rethink_mh.experiments.loop_crossfit merge \
    "${fold_args[@]}" \
    --raw-oof-dir "${RAW_OOF_DIR}" \
    --raw-artifact-manifest "${RAW_ARTIFACT_MANIFEST}" \
    --output-dir "${frozen}" \
    --expected-fold-count 5 \
    --expected-session-count 107 >/dev/null
fi

prediction_manifest="${CROSS_ROOT}/manifests/predictions.manifest.json"
if [[ ! -f "${prediction_manifest}" ]]; then
  "${PYTHON}" \
    -m rethink_mh.experiments.frozen_artifact_manifest \
    --root "${frozen}" \
    --output "${prediction_manifest}" >/dev/null
fi

evaluation="${CROSS_ROOT}/evaluation"
if [[ ! -e "${evaluation}" ]]; then
  "${PYTHON}" -m rethink_mh.experiments.loop_crossfit evaluate \
    --predictions-dir "${frozen}" \
    --predictions-artifact-manifest "${prediction_manifest}" \
    --prepared-root "${PREPARED_ROOT}" \
    --prepared-artifact-manifest "${PREPARED_ARTIFACT_MANIFEST}" \
    --config "${CONFIG_PATH}" \
    --output-dir "${evaluation}" >/dev/null
fi

touch "${CROSS_ROOT}/CROSSFIT_COMPLETE"
echo "$(date -Is) five-fold Revision-SFT crossfit evaluation complete."
