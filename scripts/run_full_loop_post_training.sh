#!/usr/bin/env bash
# Run Action-SFT, Revision-SFT, and ORPO post-training on the full-loop package.
set -euo pipefail

GPU="${GPU:?set one physical GPU index}"
PROJECT_ROOT="${PROJECT_ROOT:-/path/to/rethink-mh}"
PYTHON="${PYTHON:-python}"
MODEL_PATH="${MODEL_PATH:-/path/to/Qwen2.5-Omni-7B}"
PACKAGE_DIR="${PACKAGE_DIR:-${PROJECT_ROOT}/outputs/full_loop_training_package/daic_woz}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/full_loop_post_training/daic_woz}"
LOG_ROOT="${PROJECT_ROOT}/outputs/logs"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-full_loop_post_training}"

for required in \
  "${MODEL_PATH}" \
  "${PACKAGE_DIR}/manifest.json" \
  "${PACKAGE_DIR}/revision_sft.jsonl" \
  "${PACKAGE_DIR}/preferences_initial.jsonl" \
  "${PACKAGE_DIR}/trajectories.jsonl"; do
  if [[ ! -e "${required}" ]]; then
    echo "full-loop post-training input is missing: ${required}" >&2
    exit 3
  fi
done

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${OUTPUT_DIR}" "${LOG_ROOT}"

require_complete_or_absent() {
  local directory="$1"
  local marker="$2"
  if [[ -d "${directory}" && ! -f "${directory}/${marker}" ]]; then
    echo "partial post-training stage requires manual audit: ${directory}" >&2
    exit 4
  fi
}

LOOP_DIR="${OUTPUT_DIR}/revision_sft"
require_complete_or_absent "${LOOP_DIR}" summary.json
if [[ ! -f "${LOOP_DIR}/summary.json" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
    -m rethink_mh.experiments.rethink_post_training \
    --stage revision_sft \
    --records "${PACKAGE_DIR}/revision_sft.jsonl" \
    --model-path "${MODEL_PATH}" \
    --output-dir "${LOOP_DIR}" \
    --epochs 2 \
    --learning-rate 5e-5 \
    --gradient-accumulation 8 \
    --transition-weighting inverse_session_frequency \
    --max-effective-sample-weight 10 \
    --sampling-policy transition_balanced \
    --max-tokens 8192 \
    --experiment-name "${EXPERIMENT_NAME}" \
    --device cuda:0 \
    2>&1 | tee "${LOG_ROOT}/full_loop.revision_sft.log"
fi

ORPO_INITIAL_DIR="${OUTPUT_DIR}/orpo_initial"
require_complete_or_absent "${ORPO_INITIAL_DIR}" summary.json
if [[ ! -f "${ORPO_INITIAL_DIR}/summary.json" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
    -m rethink_mh.experiments.rethink_post_training \
    --stage orpo \
    --records "${PACKAGE_DIR}/preferences_initial.jsonl" \
    --model-path "${MODEL_PATH}" \
    --init-adapter "${LOOP_DIR}/adapter" \
    --output-dir "${ORPO_INITIAL_DIR}" \
    --epochs 2 \
    --learning-rate 2e-5 \
    --orpo-beta 0.10 \
    --orpo-forward-mode concatenated \
    --gradient-accumulation 8 \
    --transition-weighting inverse_session_frequency \
    --max-effective-sample-weight 10 \
    --max-preferences-per-session 2 \
    --constraint-session-fraction 0.20 \
    --sampling-policy transition_balanced \
    --preference-audit-limit 96 \
    --max-tokens 8192 \
    --experiment-name "${EXPERIMENT_NAME}" \
    --device cuda:0 \
    2>&1 | tee "${LOG_ROOT}/full_loop.orpo_initial.log"
fi

REFRESH_DIR="${OUTPUT_DIR}/on_policy_refresh"
require_complete_or_absent "${REFRESH_DIR}" manifest.json
if [[ ! -f "${REFRESH_DIR}/manifest.json" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
    -m rethink_mh.experiments.rethink_on_policy \
    --preferences-initial "${PACKAGE_DIR}/preferences_initial.jsonl" \
    --trajectories "${PACKAGE_DIR}/trajectories.jsonl" \
    --orpo-summary "${ORPO_INITIAL_DIR}/summary.json" \
    --model-path "${MODEL_PATH}" \
    --adapter-path "${ORPO_INITIAL_DIR}/adapter" \
    --output-dir "${REFRESH_DIR}" \
    --max-input-tokens 8192 \
    --max-new-tokens 420 \
    --max-sessions 107 \
    --minimum-acceptance-ratio 0.15 \
    --minimum-accepted-sessions 12 \
    --device cuda:0 \
    2>&1 | tee "${LOG_ROOT}/full_loop.refresh.log"
fi

"${PYTHON}" - "${REFRESH_DIR}/manifest.json" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
if manifest.get("gate", {}).get("passed") is not True:
    raise SystemExit("current-policy rollout refresh gate failed; refreshed ORPO remains blocked")
PY

ORPO_REFRESHED_DIR="${OUTPUT_DIR}/orpo_refreshed"
require_complete_or_absent "${ORPO_REFRESHED_DIR}" summary.json
if [[ ! -f "${ORPO_REFRESHED_DIR}/summary.json" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
    -m rethink_mh.experiments.rethink_post_training \
    --stage orpo \
    --records "${REFRESH_DIR}/preferences_refreshed_optimization.jsonl" \
    --model-path "${MODEL_PATH}" \
    --init-adapter "${ORPO_INITIAL_DIR}/adapter" \
    --output-dir "${ORPO_REFRESHED_DIR}" \
    --epochs 1 \
    --learning-rate 1e-5 \
    --orpo-beta 0.10 \
    --orpo-forward-mode concatenated \
    --gradient-accumulation 8 \
    --transition-weighting inverse_session_frequency \
    --max-effective-sample-weight 10 \
    --sampling-policy transition_balanced \
    --preference-audit-limit 96 \
    --max-tokens 8192 \
    --experiment-name "${EXPERIMENT_NAME}" \
    --device cuda:0 \
    2>&1 | tee "${LOG_ROOT}/full_loop.orpo_refreshed.log"
fi

touch "${OUTPUT_DIR}/PIPELINE_COMPLETE"
echo "$(date -Is) Revision-SFT, initial ORPO, rollout refresh, and refreshed ORPO complete."
