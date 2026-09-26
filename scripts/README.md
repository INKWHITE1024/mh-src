# Script map

Scripts stay in a flat layout and are grouped here by pipeline stage. Core
algorithms live in `src/rethink_mh/`; scripts only pin arguments, orchestrate
processes, and route artifacts.

## Input audit and evidence compilation

- `compile_transcript_artifacts.sh` — recompile A/V Evidence and build the compact transcript layer for DAIC-WOZ and E-DAIC.
- `run_compile_all_datasets.sh` — input audit, train-only reference fitting, and train/dev/test native Evidence compilation with split manifest snapshots.

## Orchestration

- `wait_for_gpu_and_train.sh` — poll GPU free memory and utilization, claim a card collision-safe with `flock`, then launch the selected mode.

## Initial-judgment native OOF training (DAIC-WOZ)

- `prepare_native_oof_fold.sh` — refit the fold-local reference, compile fold-local native Evidence, and audit inputs for one outer fold.
- `run_native_oof_fold.sh` — train the Evidence+transcript adapter on one outer fold with the fixed-epoch protocol.
- `run_native_oof_sequence.sh` — run the five folds, aggregate per seed, and close with the matched-OOF comparison against a frozen reference ensemble.

## Qwen label and transcript training

- `run_qwen_label_independent.sh` — train a single-seed Qwen label adapter on the official train split.
- `run_qwen_label_oof_fold.sh` — train one strict-OOF fold of the Qwen label adapter.
- `run_qwen_transcript_independent.sh` — train with Evidence plus the compact transcript layer.

## Evidence literacy

- `prepare_evidence_literacy_single_fold.sh` — build the label-free query/selection training package for one fold.
- `run_evidence_literacy_single_fold.sh` — train the literacy adapter on one fold and run the held-out contract gate.
- `prepare_evidence_literacy_oof_fold.sh` — the same package build for an arbitrary outer fold.
- `run_evidence_literacy_oof_fold.sh` — literacy training for one outer fold.
- `run_evidence_literacy_oof_sequence.sh` — run all folds and assemble the reviewer route map.

## Full loop

- `prepare_full_loop.sh` — publish physically separated label-free inference plans and outcome-only files.
- `run_full_loop_fold.sh` — collect label-free fold-local trajectories for one outer fold.
- `run_full_loop_sequence.sh` — run collection across folds and freeze the raw OOF trajectories.
- `run_full_loop_post_sequence.sh` — outcome join, fixed-arm scoring, and training-package build after the raw freeze.
- `rethink-fit-audit fit` / `rethink-fit-audit decide` — fit the learned reliability audit on frozen training-fold OOF initial assessments, fix its risk threshold on development assessments at a target trigger rate, and write label-free trigger decisions; pass them to `loop_evaluate` and `loop_crossfit collect` with `--audit-decisions`, or pass the fitted model to the online workflows with `--audit-model`. Without these flags the heuristic rules in `configs/rethinking/heuristic_baseline.json` remain as a baseline.
- `run_full_loop_post_training.sh` — Revision-SFT → ORPO initial → current-policy rollout refresh → ORPO refreshed for one fold.
- `run_loop_crossfit_fold.sh` / `run_loop_crossfit_sequence.sh` — cross-fitted evaluation of the trained revision policy.
- `run_contract_recovery_fold.sh` / `run_contract_recovery_sequence.sh` — re-run sessions that failed the output contract, then collect the full set.

## Post-training and baselines

- `run_rethink_post_training_pipeline.sh` — run the offline loop-aligned post-training stages for one dataset.
- `run_rethink_post_training_all.sh` — the reporting pass over finished post-training runs.

## D-Vlog

- `run_d_vlog_supervised.sh` — D-Vlog independent supervised protocol: preflight gates, three-seed training, freeze, and metric-only evaluation.
- `run_d_vlog_supervised_single.sh` — the gate or training step for one seed.

Check each script's referenced dataset, Evidence root, reference, label files,
GPU, and output directories before running; do not infer scope from the file
name alone.
