#!/usr/bin/env bash
# Train one fold of the native strict-OOF initial-judgment run on a chosen GPU.
set -euo pipefail

DATASET="${1:?usage: run_native_oof_fold.sh daic_woz FOLD_INDEX}"
FOLD_INDEX="${2:?usage: run_native_oof_fold.sh daic_woz FOLD_INDEX}"
PROJECT_ROOT="${PROJECT_ROOT:-/path/to/rethink-mh}"
PYTHON="${PYTHON:-python}"
MODEL_PATH="${MODEL_PATH:-/path/to/Qwen2.5-Omni-7B}"
DATA_ROOT="${DATA_ROOT:-/path/to/datasets/DAIC-WOZ}"
TRANSCRIPT_ROOT="${TRANSCRIPT_ROOT:-${PROJECT_ROOT}/artifacts/transcripts/compact/daic_woz}"
GPU="${GPU:?set GPU to one physical GPU index}"
MIN_FREE_MB="${MIN_FREE_MB:-40000}"
TRAIN_SEED="${TRAIN_SEED:-42}"
SPLIT_SEED="${SPLIT_SEED:-42}"
FOLDS="${FOLDS:-5}"
EPOCHS="${EPOCHS:-2}"
ARTIFACT_GROUP="${ARTIFACT_GROUP:-native_from_scratch}"
RUN_GROUP="${RUN_GROUP:-qwen_native_from_scratch}"

if [[ "${DATASET}" != "daic_woz" ]]; then
  echo "native locked validation currently supports daic_woz only" >&2
  exit 2
fi
if [[ ! "${FOLD_INDEX}" =~ ^[0-9]+$ ]] \
  || (( FOLD_INDEX < 0 || FOLD_INDEX >= FOLDS )); then
  echo "fold index must be in [0, $((FOLDS - 1))]" >&2
  exit 2
fi
free_mb="$(nvidia-smi --id="${GPU}" --query-gpu=memory.free --format=csv,noheader,nounits)"
if (( free_mb < MIN_FREE_MB )); then
  echo "GPU ${GPU} has ${free_mb} MiB free; ${MIN_FREE_MB} MiB is required." >&2
  exit 3
fi

TRAIN_LABELS="${DATA_ROOT}/train_split_Depression_AVEC2017.csv"
FOLD_ROOT="${PROJECT_ROOT}/artifacts/${ARTIFACT_GROUP}/daic_woz/seed${SPLIT_SEED}/fold${FOLD_INDEX}"
REFERENCE="${FOLD_ROOT}/reference.json"
EVIDENCE_ROOT="${FOLD_ROOT}/evidence/native/daic_woz"
if [[ ! -f "${FOLD_ROOT}/READY.json" ]]; then
  echo "native fold has not passed preparation: ${FOLD_ROOT}" >&2
  exit 4
fi

OUTPUT_DIR="${PROJECT_ROOT}/outputs/${RUN_GROUP}/daic_woz/seed${TRAIN_SEED}/fold${FOLD_INDEX}"
LOG="${PROJECT_ROOT}/outputs/logs/daic_woz.${RUN_GROUP}.seed${TRAIN_SEED}.fold${FOLD_INDEX}.log"
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "refusing to overwrite ${OUTPUT_DIR}" >&2
  exit 4
fi
mkdir -p "$(dirname "${OUTPUT_DIR}")" "$(dirname "${LOG}")"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export TOKENIZERS_PARALLELISM=false

"${PYTHON}" -m rethink_mh.experiments.qwen_label_baseline \
  --dataset daic_woz \
  --model-path "${MODEL_PATH}" \
  --task "${PROJECT_ROOT}/configs/tasks/phq8_depression_binary.json" \
  --evidence-root "${EVIDENCE_ROOT}" \
  --evidence-protocol native \
  --transcript-root "${TRANSCRIPT_ROOT}" \
  --train-labels "${TRAIN_LABELS}" \
  --label-column PHQ8_Score \
  --label-threshold 10 \
  --output-dir "${OUTPUT_DIR}" \
  --folds "${FOLDS}" \
  --fold-index "${FOLD_INDEX}" \
  --oof-reference "${REFERENCE}" \
  --seed "${TRAIN_SEED}" \
  --split-seed "${SPLIT_SEED}" \
  --evidence-density full \
  --max-evidence-tokens 2000 \
  --max-transcript-tokens 3000 \
  --max-input-tokens 4800 \
  --head-type binary \
  --checkpoint-metric none \
  --epochs "${EPOCHS}" \
  --batch-size 1 \
  --eval-batch-size 1 \
  --gradient-accumulation 8 \
  --learning-rate 5e-5 \
  --head-learning-rate 1e-5 \
  --scheduler-type linear \
  --warmup-ratio 0.10 \
  --weight-decay 0 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --max-grad-norm 1 \
  --device cuda:0 \
  --no-class-weighting \
  2>&1 | tee "${LOG}"

"${PYTHON}" - "${OUTPUT_DIR}/summary.json" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
summary = json.loads(path.read_text(encoding="utf-8"))
if summary.get("experiment") != "qwen_thinker_av_transcript_label_training_native":
    raise SystemExit("native training summary has the wrong experiment identity")
if summary.get("evidence_protocol") != "native":
    raise SystemExit("native training summary has the wrong Evidence protocol")
if summary.get("fit_boundaries", {}).get("test_labels_accessed") is not False:
    raise SystemExit("native training accessed test labels")
if summary.get("training", {}).get("fixed_epoch_protocol") is not True:
    raise SystemExit("native OOF did not use the fixed-epoch protocol")
PY
