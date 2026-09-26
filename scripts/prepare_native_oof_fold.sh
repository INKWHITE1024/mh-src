#!/usr/bin/env bash
# Prepare one fold's strict-OOF native evidence compilation against a frozen reference assignment.
set -euo pipefail

DATASET="${1:?usage: prepare_native_oof_fold.sh daic_woz FOLD_INDEX}"
FOLD_INDEX="${2:?usage: prepare_native_oof_fold.sh daic_woz FOLD_INDEX}"
PROJECT_ROOT="${PROJECT_ROOT:-/path/to/rethink-mh}"
PYTHON="${PYTHON:-python}"
MODEL_PATH="${MODEL_PATH:-/path/to/Qwen2.5-Omni-7B}"
DATA_ROOT="${DATA_ROOT:-/path/to/datasets/DAIC-WOZ}"
TRANSCRIPT_ROOT="${TRANSCRIPT_ROOT:-${PROJECT_ROOT}/artifacts/transcripts/compact/daic_woz}"
FROZEN_REFERENCE_ARTIFACT_ROOT="${FROZEN_REFERENCE_ARTIFACT_ROOT:?set the read-only frozen reference strict-OOF artifact root}"
SPLIT_SEED="${SPLIT_SEED:-42}"
FOLDS="${FOLDS:-5}"
WORKERS="${WORKERS:-8}"
ARTIFACT_GROUP="${ARTIFACT_GROUP:-native_from_scratch}"

if [[ "${DATASET}" != "daic_woz" ]]; then
  echo "native locked validation currently supports daic_woz only" >&2
  exit 2
fi
if [[ ! "${FOLD_INDEX}" =~ ^[0-9]+$ ]] \
  || (( FOLD_INDEX < 0 || FOLD_INDEX >= FOLDS )); then
  echo "fold index must be in [0, $((FOLDS - 1))]" >&2
  exit 2
fi

TRAIN_LABELS="${DATA_ROOT}/train_split_Depression_AVEC2017.csv"
CONFIG="${PROJECT_ROOT}/configs/textualization/daic_woz.yaml"
FROZEN_FOLD="${FROZEN_REFERENCE_ARTIFACT_ROOT}/seed${SPLIT_SEED}/fold${FOLD_INDEX}"
SOURCE_ASSIGNMENT="${FROZEN_FOLD}/assignment.csv"
SOURCE_ASSIGNMENT_MANIFEST="${FROZEN_FOLD}/assignment_manifest.json"
FOLD_ROOT="${PROJECT_ROOT}/artifacts/${ARTIFACT_GROUP}/daic_woz/seed${SPLIT_SEED}/fold${FOLD_INDEX}"
READY_MANIFEST="${FOLD_ROOT}/READY.json"

if [[ -f "${READY_MANIFEST}" ]]; then
  echo "$(date -Is) native fold is already prepared: ${FOLD_ROOT}"
  exit 0
fi
if [[ -e "${FOLD_ROOT}" ]]; then
  echo "partial native fold exists without READY.json: ${FOLD_ROOT}" >&2
  exit 4
fi
for path in \
  "${TRAIN_LABELS}" \
  "${CONFIG}" \
  "${SOURCE_ASSIGNMENT}" \
  "${SOURCE_ASSIGNMENT_MANIFEST}"; do
  if [[ ! -f "${path}" ]]; then
    echo "required preparation input is missing: ${path}" >&2
    exit 4
  fi
done
if [[ ! -d "${TRANSCRIPT_ROOT}" ]]; then
  echo "frozen transcript layer is missing: ${TRANSCRIPT_ROOT}" >&2
  exit 4
fi

mkdir -p "${FOLD_ROOT}"
ASSIGNMENT="${FOLD_ROOT}/assignment.csv"
ASSIGNMENT_MANIFEST="${FOLD_ROOT}/assignment_manifest.json"
cp "${SOURCE_ASSIGNMENT}" "${ASSIGNMENT}"
cp "${SOURCE_ASSIGNMENT_MANIFEST}" "${ASSIGNMENT_MANIFEST}"

REFERENCE="${FOLD_ROOT}/reference.json"
EVIDENCE_OUTPUT="${FOLD_ROOT}/evidence"
EVIDENCE_ROOT="${EVIDENCE_OUTPUT}/native/daic_woz"
VALIDATION_OUTPUT="${FOLD_ROOT}/input_audit"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false

"${PYTHON}" - "${ASSIGNMENT_MANIFEST}" "${DATASET}" "${SPLIT_SEED}" "${FOLD_INDEX}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
dataset, seed, fold = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
payload = json.loads(path.read_text(encoding="utf-8"))
if (
    payload.get("dataset") != dataset
    or int(payload.get("seed", -1)) != seed
    or int(payload.get("fold_index", -1)) != fold
):
    raise SystemExit(f"frozen assignment identity mismatch: {path}")
PY

"${PYTHON}" -m rethink_mh.textualization fit-reference \
  --dataset daic_woz \
  --root "${DATA_ROOT}" \
  --config "${CONFIG}" \
  --split-file "${ASSIGNMENT}" \
  --split-value fit \
  --fit-split train \
  --seed 17 \
  --output "${REFERENCE}"

"${PYTHON}" -m rethink_mh.textualization compile \
  --dataset daic_woz \
  --root "${DATA_ROOT}" \
  --config "${CONFIG}" \
  --split-file "${ASSIGNMENT}" \
  --split-name "native_train_oof_fold${FOLD_INDEX}" \
  --reference "${REFERENCE}" \
  --output-root "${EVIDENCE_OUTPUT}" \
  --protocol-version native \
  --workers "${WORKERS}"

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
  --output-dir "${VALIDATION_OUTPUT}" \
  --folds "${FOLDS}" \
  --fold-index "${FOLD_INDEX}" \
  --oof-reference "${REFERENCE}" \
  --seed "${SPLIT_SEED}" \
  --split-seed "${SPLIT_SEED}" \
  --evidence-density full \
  --max-evidence-tokens 2000 \
  --max-transcript-tokens 3000 \
  --max-input-tokens 4800 \
  --head-type binary \
  --validate-only

"${PYTHON}" - \
  "${SOURCE_ASSIGNMENT}" \
  "${ASSIGNMENT}" \
  "${SOURCE_ASSIGNMENT_MANIFEST}" \
  "${ASSIGNMENT_MANIFEST}" \
  "${REFERENCE}" \
  "${EVIDENCE_ROOT}/run_manifest.json" \
  "${VALIDATION_OUTPUT}/validation_summary.json" \
  "${READY_MANIFEST}" <<'PY'
import hashlib
import json
import pathlib
import sys

(
    source_assignment,
    assignment,
    source_manifest,
    assignment_manifest,
    reference_path,
    evidence_path,
    audit_path,
    output_path,
) = map(pathlib.Path, sys.argv[1:])

def digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

if digest(source_assignment) != digest(assignment):
    raise SystemExit("native assignment differs from frozen reference assignment")
if digest(source_manifest) != digest(assignment_manifest):
    raise SystemExit("native assignment manifest differs from frozen reference")
assignment_payload = json.loads(assignment_manifest.read_text(encoding="utf-8"))
reference = json.loads(reference_path.read_text(encoding="utf-8"))
evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
audit = json.loads(audit_path.read_text(encoding="utf-8"))
if reference["session_ids_sha256"] != assignment_payload["reference_session_ids_sha256"]:
    raise SystemExit("reference fit IDs do not match the frozen assignment")
if audit["fit_ids_sha256"] != assignment_payload["fit_ids_sha256"]:
    raise SystemExit("training fit IDs do not match the frozen assignment")
if audit["validation_ids_sha256"] != assignment_payload["holdout_ids_sha256"]:
    raise SystemExit("training holdout IDs do not match the frozen assignment")
if audit.get("oof_reference", {}).get("strict") is not True:
    raise SystemExit("strict OOF reference audit did not pass")
if audit.get("evidence_protocol") != "native":
    raise SystemExit("input audit did not use native")
payload = {
    "schema_version": "1.0.0",
    "experiment": "native_from_scratch",
    "dataset": "daic_woz",
    "evidence_protocol": "native",
    "split_seed": assignment_payload["seed"],
    "folds": assignment_payload["folds"],
    "fold_index": assignment_payload["fold_index"],
    "reference_id": reference["reference_id"],
    "fit_count": assignment_payload["fit_count"],
    "holdout_count": assignment_payload["holdout_count"],
    "prompt_max": max(
        audit["token_lengths"]["fit_max"],
        audit["token_lengths"]["validation_max"],
    ),
    "evidence_token_max": max(
        audit["token_lengths"]["by_split"]["fit"]["evidence_max"],
        audit["token_lengths"]["by_split"]["validation"]["evidence_max"],
    ),
    "frozen_reference_assignment_reused_byte_exactly": True,
    "frozen_reference_evidence_or_model_reused": False,
    "label_boundaries": {
        "assignment_label_access": "frozen_existing_partition",
        "evidence_compiler_label_access": False,
        "transcript_compiler_label_access": False,
        "dev_or_test_access": False,
    },
    "sha256": {
        path.name: digest(path)
        for path in (
            assignment,
            assignment_manifest,
            reference_path,
            evidence_path,
            audit_path,
        )
    },
}
temporary = output_path.with_name(f".{output_path.name}.tmp")
temporary.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
temporary.replace(output_path)
print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
PY
