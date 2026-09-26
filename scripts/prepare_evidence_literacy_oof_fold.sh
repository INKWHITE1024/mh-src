#!/usr/bin/env bash
# Dispatch one outer-fold Evidence Literacy preparation to the single-fold preparer.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/rethink-mh}"
FOLD_INDEX="${FOLD_INDEX:?set outer FOLD_INDEX}"
if [[ ! "${FOLD_INDEX}" =~ ^[0-9]+$ ]] || (( FOLD_INDEX < 0 || FOLD_INDEX > 4 )); then
  echo "FOLD_INDEX must be an integer from 0 to 4" >&2
  exit 2
fi

exec env \
  PROJECT_ROOT="${PROJECT_ROOT}" \
  FOLD_INDEX="${FOLD_INDEX}" \
  ARTIFACT_GROUP="${ARTIFACT_GROUP:-evidence_literacy_oof}" \
  "${PROJECT_ROOT}/scripts/prepare_evidence_literacy_single_fold.sh"
