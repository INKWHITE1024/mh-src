#!/usr/bin/env bash
# Train one full-split Qwen label baseline for a dataset on a chosen GPU.
set -euo pipefail

DATASET="${1:?usage: run_qwen_label_independent.sh daic_woz|e_daic}"
PROJECT_ROOT="${PROJECT_ROOT:-/path/to/rethink-mh}"
PYTHON="${PYTHON:-python}"
MODEL_PATH="${MODEL_PATH:-/path/to/Qwen2.5-Omni-7B}"
GPU="${GPU:?set GPU to one physical GPU index}"
MIN_FREE_MB="${MIN_FREE_MB:-40000}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-3}"
EVALUATE_LABELED_TEST="${EVALUATE_LABELED_TEST:-0}"

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
    TEST_LABELS="/path/to/datasets/DAIC-WOZ/test_split_Depression_AVEC2017.csv"
    LABEL_COLUMN="PHQ8_Score"
    ;;
  e_daic)
    EVIDENCE_ROOT="${PROJECT_ROOT}/artifacts/evidence/native/e_daic"
    TRAIN_LABELS="/path/to/datasets/E-DAIC/labels/train_split.csv"
    DEV_LABELS="/path/to/datasets/E-DAIC/labels/dev_split.csv"
    TEST_LABELS="/path/to/datasets/E-DAIC/labels/test_split.csv"
    LABEL_COLUMN="PHQ_Score"
    ;;
  *)
    echo "unsupported dataset: ${DATASET}" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="${PROJECT_ROOT}/outputs/qwen_label/${DATASET}/full_seed${SEED}"
LOG="${PROJECT_ROOT}/outputs/logs/${DATASET}.qwen_label.full_seed${SEED}.log"
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "refusing to overwrite ${OUTPUT_DIR}" >&2
  exit 4
fi
mkdir -p "$(dirname "${OUTPUT_DIR}")" "$(dirname "${LOG}")"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${GPU}"

test_access_args=()
if [[ "${EVALUATE_LABELED_TEST}" == "1" ]]; then
  test_access_args+=(--evaluate-labeled-test)
fi

"${PYTHON}" -m rethink_mh.experiments.qwen_label_baseline \
  --dataset "${DATASET}" \
  --model-path "${MODEL_PATH}" \
  --task "${PROJECT_ROOT}/configs/tasks/phq8_depression_binary.json" \
  --evidence-root "${EVIDENCE_ROOT}" \
  --train-labels "${TRAIN_LABELS}" \
  --dev-labels "${DEV_LABELS}" \
  --test-labels "${TEST_LABELS}" \
  --label-column "${LABEL_COLUMN}" \
  --label-threshold 10 \
  --output-dir "${OUTPUT_DIR}" \
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
  "${test_access_args[@]}" \
  2>&1 | tee "${LOG}"
