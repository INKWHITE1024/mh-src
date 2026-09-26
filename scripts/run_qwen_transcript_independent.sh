#!/usr/bin/env bash
# Train one transcript-text Qwen baseline for a dataset on a chosen GPU.
set -euo pipefail

DATASET="${1:?usage: run_qwen_transcript_independent.sh daic_woz|e_daic}"
PROJECT_ROOT="${PROJECT_ROOT:-/path/to/rethink-mh}"
PYTHON="${PYTHON:-python}"
MODEL_PATH="${MODEL_PATH:-/path/to/Qwen2.5-Omni-7B}"
GPU="${GPU:?set GPU to one physical GPU index}"
MIN_FREE_MB="${MIN_FREE_MB:-40000}"
SEED="${SEED:-42}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
HEAD_LEARNING_RATE="${HEAD_LEARNING_RATE:-1e-5}"
CHECKPOINT_METRIC="${CHECKPOINT_METRIC:-roc_auc}"
CLASS_WEIGHTING="${CLASS_WEIGHTING:-1}"
RUN_GROUP="${RUN_GROUP:-qwen_training_transcript}"
RUN_SUFFIX="${RUN_SUFFIX:-}"

if [[ "${CLASS_WEIGHTING}" != "0" && "${CLASS_WEIGHTING}" != "1" ]]; then
  echo "CLASS_WEIGHTING must be 0 or 1" >&2
  exit 2
fi
if [[ "${CHECKPOINT_METRIC}" != "roc_auc" \
  && "${CHECKPOINT_METRIC}" != "average_precision" \
  && "${CHECKPOINT_METRIC}" != "log_loss" ]]; then
  echo "unsupported CHECKPOINT_METRIC: ${CHECKPOINT_METRIC}" >&2
  exit 2
fi

free_mb="$(nvidia-smi --id="${GPU}" --query-gpu=memory.free --format=csv,noheader,nounits)"
if (( free_mb < MIN_FREE_MB )); then
  echo "GPU ${GPU} has ${free_mb} MiB free; ${MIN_FREE_MB} MiB is required." >&2
  exit 3
fi

case "${DATASET}" in
  daic_woz)
    EVIDENCE_ROOT="${PROJECT_ROOT}/artifacts/evidence/native/daic_woz"
    TRAIN_LABELS="/path/to/datasets/DAIC-WOZ/train_split_Depression_AVEC2017.csv"
    DEV_LABELS="/path/to/datasets/DAIC-WOZ/dev_split_Depression_AVEC2017.csv"
    LABEL_COLUMN="PHQ8_Score"
    DEFAULT_EPOCHS=8
    ;;
  e_daic)
    EVIDENCE_ROOT="${PROJECT_ROOT}/artifacts/evidence/native/e_daic"
    TRAIN_LABELS="/path/to/datasets/E-DAIC/labels/train_split.csv"
    DEV_LABELS="/path/to/datasets/E-DAIC/labels/dev_split.csv"
    LABEL_COLUMN="PHQ_Score"
    DEFAULT_EPOCHS=6
    ;;
  *)
    echo "unsupported dataset: ${DATASET}" >&2
    exit 2
    ;;
esac
EPOCHS="${EPOCHS:-${DEFAULT_EPOCHS}}"
TRANSCRIPT_ROOT="${PROJECT_ROOT}/artifacts/transcripts/compact/${DATASET}"

RUN_NAME="av_transcript_compact_binary_seed${SEED}${RUN_SUFFIX}"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/${RUN_GROUP}/${DATASET}/${RUN_NAME}"
LOG="${PROJECT_ROOT}/outputs/logs/${DATASET}.${RUN_GROUP}.${RUN_NAME}.log"
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "refusing to overwrite ${OUTPUT_DIR}" >&2
  exit 4
fi
mkdir -p "$(dirname "${OUTPUT_DIR}")" "$(dirname "${LOG}")"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export TOKENIZERS_PARALLELISM=false

CLASS_WEIGHTING_ARGS=()
if [[ "${CLASS_WEIGHTING}" == "0" ]]; then
  CLASS_WEIGHTING_ARGS+=(--no-class-weighting)
fi

"${PYTHON}" -m rethink_mh.experiments.qwen_label_baseline \
  --dataset "${DATASET}" \
  --model-path "${MODEL_PATH}" \
  --task "${PROJECT_ROOT}/configs/tasks/phq8_depression_binary.json" \
  --evidence-root "${EVIDENCE_ROOT}" \
  --transcript-root "${TRANSCRIPT_ROOT}" \
  --train-labels "${TRAIN_LABELS}" \
  --dev-labels "${DEV_LABELS}" \
  --label-column "${LABEL_COLUMN}" \
  --label-threshold 10 \
  --output-dir "${OUTPUT_DIR}" \
  --seed "${SEED}" \
  --evidence-density compact \
  --max-evidence-tokens 2000 \
  --max-transcript-tokens 3000 \
  --max-input-tokens 4800 \
  --head-type binary \
  --checkpoint-metric "${CHECKPOINT_METRIC}" \
  --epochs "${EPOCHS}" \
  --batch-size 1 \
  --eval-batch-size 1 \
  --gradient-accumulation 8 \
  --learning-rate "${LEARNING_RATE}" \
  --head-learning-rate "${HEAD_LEARNING_RATE}" \
  --scheduler-type linear \
  --warmup-ratio 0.10 \
  --weight-decay 0 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --max-grad-norm 1 \
  --device cuda:0 \
  ${CLASS_WEIGHTING_ARGS[@]+"${CLASS_WEIGHTING_ARGS[@]}"} \
  2>&1 | tee "${LOG}"
