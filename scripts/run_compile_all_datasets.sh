#!/usr/bin/env bash
# Compile native evidence for all datasets and snapshot split manifests.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/rethink-mh}"
PYTHON="${PYTHON:-python}"
DAIC_ROOT="${DAIC_ROOT:-/path/to/datasets/DAIC-WOZ}"
EDAIC_ROOT="${EDAIC_ROOT:-/path/to/datasets/E-DAIC}"
WORKERS="${WORKERS:-8}"

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

CONFIG_ROOT="${PROJECT_ROOT}/configs/textualization"
ARTIFACT_ROOT="${PROJECT_ROOT}/artifacts"

snapshot_split_manifests() {
  local dataset_key="$1"
  local split="$2"
  local dataset_root="${ARTIFACT_ROOT}/evidence/native/${dataset_key}"

  cp "${dataset_root}/sessions_manifest.jsonl" \
    "${dataset_root}/sessions_manifest.${split}.jsonl"
  cp "${dataset_root}/run_manifest.json" \
    "${dataset_root}/run_manifest.${split}.json"
}

"${PYTHON}" -m rethink_mh.textualization audit \
  --dataset daic_woz \
  --root "${DAIC_ROOT}" \
  --config "${CONFIG_ROOT}/daic_woz.yaml" \
  --output "${ARTIFACT_ROOT}/audits/daic_woz.json"

"${PYTHON}" -m rethink_mh.textualization fit-reference \
  --dataset daic_woz \
  --root "${DAIC_ROOT}" \
  --config "${CONFIG_ROOT}/daic_woz.yaml" \
  --split-file "${DAIC_ROOT}/train_split_Depression_AVEC2017.csv" \
  --fit-split train \
  --output "${ARTIFACT_ROOT}/references/daic_woz.train.json"

for split in train dev test; do
  "${PYTHON}" -m rethink_mh.textualization compile \
    --dataset daic_woz \
    --root "${DAIC_ROOT}" \
    --config "${CONFIG_ROOT}/daic_woz.yaml" \
    --split-file "${DAIC_ROOT}/${split}_split_Depression_AVEC2017.csv" \
    --split-name "${split}" \
    --reference "${ARTIFACT_ROOT}/references/daic_woz.train.json" \
    --output-root "${ARTIFACT_ROOT}/evidence" \
    --protocol-version native \
    --workers "${WORKERS}"
  snapshot_split_manifests daic_woz "${split}"
done

"${PYTHON}" -m rethink_mh.textualization audit \
  --dataset e_daic \
  --root "${EDAIC_ROOT}" \
  --config "${CONFIG_ROOT}/e_daic.yaml" \
  --output "${ARTIFACT_ROOT}/audits/e_daic.json"

"${PYTHON}" -m rethink_mh.textualization fit-reference \
  --dataset e_daic \
  --root "${EDAIC_ROOT}" \
  --config "${CONFIG_ROOT}/e_daic.yaml" \
  --split-file "${EDAIC_ROOT}/labels/train_split.csv" \
  --fit-split train \
  --output "${ARTIFACT_ROOT}/references/e_daic.train.json"

for split in train dev test; do
  "${PYTHON}" -m rethink_mh.textualization compile \
    --dataset e_daic \
    --root "${EDAIC_ROOT}" \
    --config "${CONFIG_ROOT}/e_daic.yaml" \
    --split-file "${EDAIC_ROOT}/labels/${split}_split.csv" \
    --split-name "${split}" \
    --reference "${ARTIFACT_ROOT}/references/e_daic.train.json" \
    --output-root "${ARTIFACT_ROOT}/evidence" \
    --protocol-version native \
    --workers "${WORKERS}"
  snapshot_split_manifests e_daic "${split}"
done
