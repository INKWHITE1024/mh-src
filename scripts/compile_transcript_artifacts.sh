#!/usr/bin/env bash
# Compile transcript-native E-DAIC evidence artifacts with an ASR-aligned audio layer.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/rethink-mh}"
PYTHON="${PYTHON:-python}"
DAIC_ROOT="${DAIC_ROOT:-/path/to/datasets/DAIC-WOZ}"
EDAIC_ROOT="${EDAIC_ROOT:-/path/to/datasets/E-DAIC}"
WORKERS="${WORKERS:-8}"

CONFIG_ROOT="${PROJECT_ROOT}/configs/textualization"
DAIC_EVIDENCE="${PROJECT_ROOT}/artifacts/evidence/native/daic_woz"
EDAIC_EVIDENCE_ROOT="${PROJECT_ROOT}/artifacts/evidence_transcript"
EDAIC_EVIDENCE="${EDAIC_EVIDENCE_ROOT}/native/e_daic"
TRANSCRIPT_ROOT="${PROJECT_ROOT}/artifacts/transcripts"
EDAIC_REFERENCE="${PROJECT_ROOT}/artifacts/references_transcript/e_daic.train.json"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# E-DAIC transcripts have no speaker field. Rebuild its A/V layer in a new root so
# ASR-aligned audio is described as mixed-speaker transcribed speech, never participant speech.
"${PYTHON}" -m rethink_mh.textualization fit-reference \
  --dataset e_daic \
  --root "${EDAIC_ROOT}" \
  --config "${CONFIG_ROOT}/e_daic.yaml" \
  --split-file "${EDAIC_ROOT}/labels/train_split.csv" \
  --fit-split train \
  --output "${EDAIC_REFERENCE}"

for split in train dev test; do
  "${PYTHON}" -m rethink_mh.textualization compile \
    --dataset e_daic \
    --root "${EDAIC_ROOT}" \
    --config "${CONFIG_ROOT}/e_daic.yaml" \
    --split-file "${EDAIC_ROOT}/labels/${split}_split.csv" \
    --split-name "${split}" \
    --reference "${EDAIC_REFERENCE}" \
    --output-root "${EDAIC_EVIDENCE_ROOT}" \
    --protocol-version native \
    --workers "${WORKERS}"
  cp "${EDAIC_EVIDENCE}/sessions_manifest.jsonl" \
    "${EDAIC_EVIDENCE}/sessions_manifest.${split}.jsonl"
  cp "${EDAIC_EVIDENCE}/run_manifest.json" \
    "${EDAIC_EVIDENCE}/run_manifest.${split}.json"
done

"${PYTHON}" -m rethink_mh.textualization compile-transcript \
  --dataset daic_woz \
  --root "${DAIC_ROOT}" \
  --config "${CONFIG_ROOT}/daic_woz.yaml" \
  --evidence-root "${DAIC_EVIDENCE}" \
  --output-root "${TRANSCRIPT_ROOT}" \
  --overwrite

"${PYTHON}" -m rethink_mh.textualization compile-transcript \
  --dataset e_daic \
  --root "${EDAIC_ROOT}" \
  --config "${CONFIG_ROOT}/e_daic.yaml" \
  --evidence-root "${EDAIC_EVIDENCE}" \
  --output-root "${TRANSCRIPT_ROOT}" \
  --overwrite

echo "Compact transcript and corrected E-DAIC A/V artifacts are complete."
