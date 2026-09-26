#!/usr/bin/env bash
# Run offline post-training (Action-SFT, Revision-SFT, ORPO) for one dataset.
set -euo pipefail

DATASET="${1:?usage: run_rethink_post_training_pipeline.sh daic_woz|e_daic|d_vlog}"
GPU="${GPU:?set GPU to one physical GPU index}"
PROJECT_ROOT="${PROJECT_ROOT:-/path/to/rethink-mh}"
PYTHON="${PYTHON:-python}"
MODEL_PATH="${MODEL_PATH:-/path/to/Qwen2.5-Omni-7B}"
MIN_FREE_MB="${MIN_FREE_MB:-40000}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-rethink_revision_post_training_v2_balanced}"
PACKAGE_ROOT="${PACKAGE_ROOT:-${PROJECT_ROOT}/outputs/rethink_post_training_v1}"
ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/rethink_post_training_v2_balanced}"
PACKAGE="${PACKAGE_ROOT}/trajectories/${DATASET}"
PREFERENCE_PACKAGE="${PACKAGE_ROOT}/preference_optimization_v1/${DATASET}"
RUN_ROOT="${ROOT}/training/${DATASET}"
LOG_ROOT="${ROOT}/logs/${DATASET}"

case "${DATASET}" in
  daic_woz|e_daic)
    LOOP_EPOCHS=2
    ORPO_INITIAL_EPOCHS=2
    REFRESH_LIMIT=192
    ;;
  d_vlog)
    LOOP_EPOCHS=2
    ORPO_INITIAL_EPOCHS=1
    REFRESH_LIMIT=192
    ;;
  *)
    echo "unsupported dataset: ${DATASET}" >&2
    exit 2
    ;;
esac

if [[ "${DATASET}" == "daic_woz" ]]; then
  ORPO_FORWARD_MODE=concatenated
else
  # DAIC reaches about 45.2/46 GB with a two-sequence forward.  The longer
  # E-DAIC and D-Vlog prompts use the exact streaming-gradient equivalent.
  ORPO_FORWARD_MODE=streaming
fi

for required in \
  "${PACKAGE}/revision_sft.jsonl" \
  "${PACKAGE}/preferences_initial.jsonl" \
  "${PACKAGE}/trajectories.jsonl" \
  "${PREFERENCE_PACKAGE}/preferences_optimization.jsonl"; do
  if [[ ! -f "${required}" ]]; then
    echo "required frozen record file is missing: ${required}" >&2
    exit 3
  fi
done

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

wait_for_gpu() {
  while true; do
    free_mb="$(nvidia-smi --id="${GPU}" --query-gpu=memory.free --format=csv,noheader,nounits)"
    if (( free_mb >= MIN_FREE_MB )); then
      return
    fi
    echo "$(date -Is) waiting: GPU ${GPU} has ${free_mb} MiB free" >&2
    sleep 20
  done
}

require_complete_or_absent() {
  stage_dir="$1"
  if [[ -d "${stage_dir}" && ! -f "${stage_dir}/summary.json" ]]; then
    echo "incomplete stage directory requires manual audit: ${stage_dir}" >&2
    exit 4
  fi
}

LOOP_DIR="${RUN_ROOT}/revision_sft"
require_complete_or_absent "${LOOP_DIR}"
if [[ ! -f "${LOOP_DIR}/summary.json" ]]; then
  wait_for_gpu
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
    -m rethink_mh.experiments.rethink_post_training \
    --stage revision_sft \
    --records "${PACKAGE}/revision_sft.jsonl" \
    --model-path "${MODEL_PATH}" \
    --output-dir "${LOOP_DIR}" \
    --epochs "${LOOP_EPOCHS}" \
    --learning-rate 5e-5 \
    --gradient-accumulation 8 \
    --transition-weighting inverse_session_frequency \
    --max-effective-sample-weight 10 \
    --experiment-name "${EXPERIMENT_NAME}" \
    --max-tokens 8192 \
    --device cuda:0 \
    2>&1 | tee "${LOG_ROOT}/revision_sft.log"
fi

ORPO_INITIAL_DIR="${RUN_ROOT}/orpo_initial"
require_complete_or_absent "${ORPO_INITIAL_DIR}"
if [[ ! -f "${ORPO_INITIAL_DIR}/summary.json" ]]; then
  wait_for_gpu
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
    -m rethink_mh.experiments.rethink_post_training \
    --stage orpo \
    --records "${PREFERENCE_PACKAGE}/preferences_optimization.jsonl" \
    --model-path "${MODEL_PATH}" \
    --init-adapter "${LOOP_DIR}/adapter" \
    --output-dir "${ORPO_INITIAL_DIR}" \
    --epochs "${ORPO_INITIAL_EPOCHS}" \
    --learning-rate 2e-5 \
    --orpo-beta 0.10 \
    --orpo-forward-mode "${ORPO_FORWARD_MODE}" \
    --gradient-accumulation 8 \
    --transition-weighting inverse_session_frequency \
    --max-effective-sample-weight 10 \
    --experiment-name "${EXPERIMENT_NAME}" \
    --preference-audit-limit 96 \
    --max-tokens 8192 \
    --device cuda:0 \
    2>&1 | tee "${LOG_ROOT}/orpo_initial.log"
fi

REFRESH_DIR="${RUN_ROOT}/on_policy_refresh"
if [[ -d "${REFRESH_DIR}" && ! -f "${REFRESH_DIR}/manifest.json" ]]; then
  echo "incomplete refresh directory requires manual audit: ${REFRESH_DIR}" >&2
  exit 5
fi
if [[ ! -f "${REFRESH_DIR}/manifest.json" ]]; then
  wait_for_gpu
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
    -m rethink_mh.experiments.rethink_on_policy \
    --preferences-initial "${PREFERENCE_PACKAGE}/preferences_optimization.jsonl" \
    --trajectories "${PACKAGE}/trajectories.jsonl" \
    --orpo-summary "${ORPO_INITIAL_DIR}/summary.json" \
    --model-path "${MODEL_PATH}" \
    --adapter-path "${ORPO_INITIAL_DIR}/adapter" \
    --output-dir "${REFRESH_DIR}" \
    --max-input-tokens 8192 \
    --max-new-tokens 420 \
    --max-sessions "${REFRESH_LIMIT}" \
    --minimum-acceptance-ratio 0.15 \
    --minimum-accepted-sessions 12 \
    --device cuda:0 \
    2>&1 | tee "${LOG_ROOT}/on_policy_refresh.log"
fi

"${PYTHON}" - "${REFRESH_DIR}/manifest.json" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
if manifest["gate"]["passed"] is not True:
    raise SystemExit("current-policy rollout refresh gate failed; ORPO refreshed remains blocked")
PY

ORPO_REFRESHED_DIR="${RUN_ROOT}/orpo_refreshed"
require_complete_or_absent "${ORPO_REFRESHED_DIR}"
if [[ ! -f "${ORPO_REFRESHED_DIR}/summary.json" ]]; then
  wait_for_gpu
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
    --orpo-forward-mode "${ORPO_FORWARD_MODE}" \
    --gradient-accumulation 8 \
    --transition-weighting inverse_session_frequency \
    --max-effective-sample-weight 10 \
    --experiment-name "${EXPERIMENT_NAME}" \
    --preference-audit-limit 96 \
    --max-tokens 8192 \
    --device cuda:0 \
    2>&1 | tee "${LOG_ROOT}/orpo_refreshed.log"
fi

"${PYTHON}" - "${DATASET}" "${PACKAGE}/manifest.json" "${RUN_ROOT}/grpo_gate.json" <<'PY'
import json
import sys
from pathlib import Path
from rethink_mh.experiments.rethink_on_policy import grpo_gate_report

dataset, package_path, output_path = sys.argv[1:]
package = json.load(open(package_path, encoding="utf-8"))
report = grpo_gate_report(dataset=dataset, trajectory_count=package["session_count"])
Path(output_path).write_text(
    json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

touch "${RUN_ROOT}/PIPELINE_COMPLETE"
echo "$(date -Is) ${DATASET} Revision-SFT -> ORPO initial -> rollout refresh -> ORPO refreshed complete"
