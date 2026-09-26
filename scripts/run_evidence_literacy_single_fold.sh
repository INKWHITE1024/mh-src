#!/usr/bin/env bash
# Run one fold's Evidence Literacy SFT training and held-out contract gate.
set -euo pipefail

GPU="${GPU:?set one physical GPU index}"
PROJECT_ROOT="${PROJECT_ROOT:-/path/to/rethink-mh}"
PYTHON="${PYTHON:-python}"
MODEL_PATH="${MODEL_PATH:-/path/to/Qwen2.5-Omni-7B}"
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:?set the frozen initial-judgment run root}"
ARTIFACT_GROUP="${ARTIFACT_GROUP:-evidence_literacy_sft}"
RUN_GROUP="${RUN_GROUP:-evidence_literacy_single_fold}"
MANIFEST_GROUP="${MANIFEST_GROUP:-evidence_literacy_single_fold_manifests}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-evidence_literacy_single_fold}"
SEED="${SEED:-42}"
FOLD_INDEX="${FOLD_INDEX:-0}"

PACKAGE_ROOT="${PROJECT_ROOT}/artifacts/${ARTIFACT_GROUP}/daic_woz/seed${SEED}/fold${FOLD_INDEX}"
SOURCE_ADAPTER="${SOURCE_RUN_ROOT}/outputs/qwen_native_from_scratch/daic_woz/seed${SEED}/fold${FOLD_INDEX}/adapter"
RUN_ROOT="${PROJECT_ROOT}/outputs/${RUN_GROUP}/daic_woz/seed${SEED}/fold${FOLD_INDEX}"
TRAIN_DIR="${RUN_ROOT}/training"
EVAL_DIR="${RUN_ROOT}/heldout_contract_gate"
LOG_ROOT="${PROJECT_ROOT}/outputs/logs"
MANIFEST_ROOT="${PROJECT_ROOT}/outputs/${MANIFEST_GROUP}/daic_woz/seed${SEED}/fold${FOLD_INDEX}"

for required in \
  "${PACKAGE_ROOT}/fit/records.jsonl" \
  "${PACKAGE_ROOT}/holdout/records.jsonl" \
  "${SOURCE_ADAPTER}/adapter_config.json" \
  "${MODEL_PATH}"; do
  if [[ ! -e "${required}" ]]; then
    echo "required literacy input is missing: ${required}" >&2
    exit 3
  fi
done
if [[ -d "${TRAIN_DIR}" && ! -f "${TRAIN_DIR}/summary.json" ]]; then
  echo "partial literacy training requires manual audit: ${TRAIN_DIR}" >&2
  exit 4
fi
if [[ -d "${EVAL_DIR}" && ! -f "${EVAL_DIR}/summary.json" ]]; then
  echo "partial literacy evaluation requires manual audit: ${EVAL_DIR}" >&2
  exit 4
fi

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}" "${MANIFEST_ROOT}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ ! -f "${TRAIN_DIR}/summary.json" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
    -m rethink_mh.experiments.rethink_post_training \
    --stage evidence_literacy_sft \
    --records "${PACKAGE_ROOT}/fit/records.jsonl" \
    --model-path "${MODEL_PATH}" \
    --init-adapter "${SOURCE_ADAPTER}" \
    --output-dir "${TRAIN_DIR}" \
    --epochs 2 \
    --learning-rate 2e-5 \
    --gradient-accumulation 8 \
    --max-tokens 6000 \
    --seed "${SEED}" \
    --experiment-name "${EXPERIMENT_NAME}" \
    --device cuda:0 \
    2>&1 | tee "${LOG_ROOT}/evidence_literacy_sft.seed${SEED}.fold${FOLD_INDEX}.train.log"
fi

if [[ ! -f "${EVAL_DIR}/summary.json" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
    -m rethink_mh.experiments.evidence_literacy_evaluate \
    --package-dir "${PACKAGE_ROOT}/holdout" \
    --model-path "${MODEL_PATH}" \
    --adapter-path "${TRAIN_DIR}/adapter" \
    --output-dir "${EVAL_DIR}" \
    --device cuda:0 \
    --minimum-query-grounded-rate 0.90 \
    --minimum-selection-grounded-rate 0.90 \
    2>&1 | tee "${LOG_ROOT}/evidence_literacy_sft.seed${SEED}.fold${FOLD_INDEX}.eval.log"
fi

for name in training heldout_contract_gate; do
  root="${RUN_ROOT}/${name}"
  manifest="${MANIFEST_ROOT}/${name}.manifest.json"
  if [[ -f "${manifest}" ]]; then
    "${PYTHON}" -m rethink_mh.experiments.frozen_artifact_manifest \
      --root "${root}" \
      --verify-manifest "${manifest}" >/dev/null
  else
    "${PYTHON}" -m rethink_mh.experiments.frozen_artifact_manifest \
      --root "${root}" \
      --output "${manifest}" >/dev/null
  fi
done

touch "${RUN_ROOT}/PIPELINE_COMPLETE"
"${PYTHON}" - "${EVAL_DIR}/summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
print(json.dumps(summary["gate"], indent=2, sort_keys=True))
PY
echo "$(date -Is) Evidence Literacy seed=${SEED} fold=${FOLD_INDEX} pilot complete."
