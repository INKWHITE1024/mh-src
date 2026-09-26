#!/usr/bin/env bash
# Prepare one fold's Evidence Literacy training and holdout packages from a frozen initial-judgment run.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/rethink-mh}"
PYTHON="${PYTHON:-python}"
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:?set the frozen initial-judgment run root}"
ARTIFACT_GROUP="${ARTIFACT_GROUP:-evidence_literacy_sft}"
SEED="${SEED:-42}"
FOLD_INDEX="${FOLD_INDEX:-0}"

SOURCE_FOLD="${SOURCE_RUN_ROOT}/artifacts/native_from_scratch/daic_woz/seed42/fold${FOLD_INDEX}"
SOURCE_OUTPUT="${SOURCE_RUN_ROOT}/outputs/qwen_native_from_scratch/daic_woz/seed${SEED}/fold${FOLD_INDEX}"
COMPARISON_DIR="${SOURCE_RUN_ROOT}/outputs/matched_oof_comparison/daic_woz"
COMPARISON_MANIFEST="${SOURCE_RUN_ROOT}/outputs/matched_oof_comparison_manifests/daic_woz/comparison.manifest.json"
TASK="${PROJECT_ROOT}/configs/tasks/phq8_depression_binary.json"
PACKAGE_ROOT="${PROJECT_ROOT}/artifacts/${ARTIFACT_GROUP}/daic_woz/seed${SEED}/fold${FOLD_INDEX}"
ASSIGNMENT="${SOURCE_FOLD}/assignment.csv"
EVIDENCE_ROOT="${SOURCE_FOLD}/evidence/native/daic_woz"

for required in \
  "${PROJECT_ROOT}" \
  "${TASK}" \
  "${ASSIGNMENT}" \
  "${EVIDENCE_ROOT}" \
  "${SOURCE_OUTPUT}/adapter/adapter_config.json" \
  "${COMPARISON_DIR}/summary.json" \
  "${COMPARISON_MANIFEST}"; do
  if [[ ! -e "${required}" ]]; then
    echo "required literacy input is missing: ${required}" >&2
    exit 3
  fi
done

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON}" -m rethink_mh.experiments.frozen_artifact_manifest \
  --root "${COMPARISON_DIR}" \
  --verify-manifest "${COMPARISON_MANIFEST}" >/dev/null

"${PYTHON}" - "${COMPARISON_DIR}/summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
gate = summary.get("preregistered_gate", {})
boundaries = summary.get("fit_boundaries", {})
if not isinstance(gate, dict):
    raise SystemExit("frozen native comparison omitted its preregistered gate")
if any(
    (
        boundaries.get("reference_retrained") is not False,
        boundaries.get("reference_recompiled") is not False,
        boundaries.get("test_accessed") is not False,
    )
):
    raise SystemExit("comparison fit boundaries are unsafe")
PY

verify_package() {
  local package="$1"
  "${PYTHON}" - "${package}/manifest.json" "${package}/records.jsonl" <<'PY'
import hashlib
import json
import sys

manifest_path, records_path = sys.argv[1:]
manifest = json.load(open(manifest_path, encoding="utf-8"))
expected = manifest["files"]["records.jsonl"]["sha256"]
observed = hashlib.sha256(open(records_path, "rb").read()).hexdigest()
if observed != expected:
    raise SystemExit("literacy package records changed after freezing")
if manifest["fit_boundaries"]["sample_targets_accessed"] is not False:
    raise SystemExit("literacy package accessed sample targets")
PY
  "${PYTHON}" -m rethink_mh.experiments.rethink_post_training \
    --stage evidence_literacy_sft \
    --records "${package}/records.jsonl" \
    --audit-only >/dev/null
}

for partition in fit holdout; do
  package="${PACKAGE_ROOT}/${partition}"
  if [[ -e "${package}" ]]; then
    if [[ ! -f "${package}/manifest.json" || ! -f "${package}/records.jsonl" ]]; then
      echo "partial literacy package requires manual audit: ${package}" >&2
      exit 4
    fi
    verify_package "${package}"
    echo "$(date -Is) verified existing literacy package: ${partition}"
    continue
  fi
  "${PYTHON}" -m rethink_mh.experiments.evidence_literacy \
    --dataset daic_woz \
    --task "${TASK}" \
    --assignment "${ASSIGNMENT}" \
    --evidence-root "${EVIDENCE_ROOT}" \
    --output-dir "${package}" \
    --partition "${partition}" \
    --maximum-queries-per-session 2 \
    --query-budget 2
  verify_package "${package}"
done

echo "$(date -Is) label-free Evidence Literacy fit/holdout packages are ready."
