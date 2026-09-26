#!/usr/bin/env bash
# Train and evaluate one cross-fitted revision-policy fold on a chosen GPU.
set -euo pipefail

GPU="${GPU:?set one physical GPU index}"
FOLD_INDEX="${FOLD_INDEX:?set outer FOLD_INDEX}"
PROJECT_ROOT="${PROJECT_ROOT:-/path/to/rethink-mh}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}}"
PYTHON="${PYTHON:-python}"
MODEL_PATH="${MODEL_PATH:-/path/to/Qwen2.5-Omni-7B}"
CROSS_ROOT="${CROSS_ROOT:-${RUN_ROOT}/outputs/loop_crossfit}"
RAW_OOF_DIR="${RAW_OOF_DIR:-${RUN_ROOT}/outputs/contract_recovery_raw_oof/daic_woz/raw_oof_frozen}"
RAW_ARTIFACT_MANIFEST="${RAW_ARTIFACT_MANIFEST:-${RUN_ROOT}/outputs/contract_recovery_manifests/raw_oof.manifest.json}"
PREPARED_INFERENCE_DIR="${PREPARED_INFERENCE_DIR:?set PREPARED_INFERENCE_DIR}"
TASK_PATH="${TASK_PATH:-${PROJECT_ROOT}/configs/tasks/phq8_depression_binary.json}"
PACKAGE_DIR="${CROSS_ROOT}/packages/fold${FOLD_INDEX}"
PACKAGE_ARTIFACT_MANIFEST="${CROSS_ROOT}/manifests/package.fold${FOLD_INDEX}.manifest.json"
TRAINING_DIR="${CROSS_ROOT}/training/fold${FOLD_INDEX}/revision_sft"
PREDICTION_DIR="${CROSS_ROOT}/predictions/fold${FOLD_INDEX}"
LOG_ROOT="${CROSS_ROOT}/logs"
EXPERIMENT_NAME="loop_crossfit_fold${FOLD_INDEX}"

if [[ ! "${FOLD_INDEX}" =~ ^[0-9]+$ ]] \
  || (( FOLD_INDEX < 0 || FOLD_INDEX > 4 )); then
  echo "FOLD_INDEX must be an integer from 0 to 4" >&2
  exit 2
fi
for required in \
  "${MODEL_PATH}" \
  "${RAW_OOF_DIR}/RAW_OOF_FROZEN" \
  "${RAW_ARTIFACT_MANIFEST}" \
  "${PREPARED_INFERENCE_DIR}/manifest.json" \
  "${PREPARED_INFERENCE_DIR}/inference_plan.jsonl" \
  "${PACKAGE_DIR}/manifest.json" \
  "${PACKAGE_DIR}/revision_sft.jsonl" \
  "${PACKAGE_ARTIFACT_MANIFEST}" \
  "${TASK_PATH}"; do
  if [[ ! -e "${required}" ]]; then
    echo "crossfit fold input is missing: ${required}" >&2
    exit 3
  fi
done
if [[ -d "${TRAINING_DIR}" && ! -f "${TRAINING_DIR}/summary.json" ]]; then
  echo "partial crossfit training requires manual audit: ${TRAINING_DIR}" >&2
  exit 4
fi
if [[ -d "${PREDICTION_DIR}" \
  && ! -f "${PREDICTION_DIR}/PIPELINE_COMPLETE" ]]; then
  echo "partial crossfit prediction requires manual audit: ${PREDICTION_DIR}" >&2
  exit 4
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${LOG_ROOT}" "$(dirname "${TRAINING_DIR}")"

if [[ ! -f "${TRAINING_DIR}/summary.json" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
    -m rethink_mh.experiments.rethink_post_training \
    --stage revision_sft \
    --records "${PACKAGE_DIR}/revision_sft.jsonl" \
    --model-path "${MODEL_PATH}" \
    --output-dir "${TRAINING_DIR}" \
    --epochs 2 \
    --learning-rate 5e-5 \
    --gradient-accumulation 8 \
    --transition-weighting inverse_session_frequency \
    --max-effective-sample-weight 10 \
    --sampling-policy transition_balanced \
    --max-tokens 8192 \
    --experiment-name "${EXPERIMENT_NAME}" \
    --device cuda:0 \
    2>&1 | tee \
      "${LOG_ROOT}/fold${FOLD_INDEX}.revision_sft.log"
fi

if [[ ! -f "${PREDICTION_DIR}/PIPELINE_COMPLETE" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
    -m rethink_mh.experiments.loop_crossfit collect \
    --raw-oof-dir "${RAW_OOF_DIR}" \
    --raw-artifact-manifest "${RAW_ARTIFACT_MANIFEST}" \
    --prepared-inference-dir "${PREPARED_INFERENCE_DIR}" \
    --training-package-dir "${PACKAGE_DIR}" \
    --training-package-artifact-manifest \
      "${PACKAGE_ARTIFACT_MANIFEST}" \
    --training-summary "${TRAINING_DIR}/summary.json" \
    --task "${TASK_PATH}" \
    --output-dir "${PREDICTION_DIR}" \
    --fold-index "${FOLD_INDEX}" \
    --device cuda:0 \
    --dtype bfloat16 \
    --attention-implementation sdpa \
    --max-new-tokens 420 \
    --max-input-tokens 8192 \
    --max-contract-retries 2 \
    2>&1 | tee \
      "${LOG_ROOT}/fold${FOLD_INDEX}.predict.log"
fi

echo "$(date -Is) crossfit fold ${FOLD_INDEX} training and prediction complete."
