#!/usr/bin/env bash
# Train one fold of the Qwen label OOF baseline on a chosen GPU.
set -euo pipefail

DATASET="${1:?usage: run_qwen_label_oof_fold.sh daic_woz|e_daic FOLD_INDEX}"
FOLD_INDEX="${2:?usage: run_qwen_label_oof_fold.sh daic_woz|e_daic FOLD_INDEX}"
PROJECT_ROOT="${PROJECT_ROOT:-/path/to/rethink-mh}"
PYTHON="${PYTHON:-python}"
MODEL_PATH="${MODEL_PATH:-/path/to/Qwen2.5-Omni-7B}"
GPU="${GPU:?set GPU to one physical GPU index}"
MIN_FREE_MB="${MIN_FREE_MB:-40000}"
SEED="${SEED:-42}"
FOLDS="${FOLDS:-5}"
EPOCHS="${EPOCHS:-3}"

if (( FOLD_INDEX < 0 || FOLD_INDEX >= FOLDS )); then
  echo "fold index must be in [0, $((FOLDS - 1))]" >&2
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
    LABEL_COLUMN="PHQ8_Score"
    ;;
  e_daic)
    EVIDENCE_ROOT="${PROJECT_ROOT}/artifacts/evidence/native/e_daic"
    TRAIN_LABELS="/path/to/datasets/E-DAIC/labels/train_split.csv"
    LABEL_COLUMN="PHQ_Score"
    ;;
  *)
    echo "unsupported dataset: ${DATASET}" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="${PROJECT_ROOT}/outputs/qwen_label/${DATASET}/oof_seed${SEED}/fold${FOLD_INDEX}"
LOG="${PROJECT_ROOT}/outputs/logs/${DATASET}.qwen_label.oof_seed${SEED}.fold${FOLD_INDEX}.log"
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "refusing to overwrite ${OUTPUT_DIR}" >&2
  exit 4
fi
mkdir -p "$(dirname "${OUTPUT_DIR}")" "$(dirname "${LOG}")"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${GPU}"

"${PYTHON}" -m rethink_mh.experiments.qwen_label_baseline \
  --dataset "${DATASET}" \
  --model-path "${MODEL_PATH}" \
  --task "${PROJECT_ROOT}/configs/tasks/phq8_depression_binary.json" \
  --evidence-root "${EVIDENCE_ROOT}" \
  --train-labels "${TRAIN_LABELS}" \
  --label-column "${LABEL_COLUMN}" \
  --label-threshold 10 \
  --output-dir "${OUTPUT_DIR}" \
  --folds "${FOLDS}" \
  --fold-index "${FOLD_INDEX}" \
  --seed "${SEED}" \
  --epochs "${EPOCHS}" \
  --batch-size 1 \
  --eval-batch-size 1 \
  --gradient-accumulation 8 \
  --learning-rate 1e-4 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --max-input-tokens 7500 \
  --device cuda:0 \
  2>&1 | tee "${LOG}"
