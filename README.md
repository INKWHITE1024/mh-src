# RETHINK-MH

Research code for RETHINK-MH: provenance-preserving multimodal evidence
textualization and a two-pass "draft → targeted evidence re-query → revise or
refer" rethinking loop for questionnaire-based depression screening, built
around Qwen2.5-Omni-7B adapters.

## Repository layout

- `src/rethink_mh/` — core package:
  - `textualization/` — compiles released DAIC-WOZ / E-DAIC / D-Vlog feature
    arrays into provenance-tracked, source-aware text units (the `native`
    protocol), fits train-only references, and renders budgeted first-pass
    session views with an atomic-window index for targeted retrieval;
  - `rethinking/` — two-pass contracts, the evidence-grounded workflow, the
    budgeted evidence agent with a tamper-evident access ledger, the
    query-guided atomic retriever, and the learned reliability audit
    (`audit.py`: an L2 logistic regression over confidence, source
    reliability, validity-weighted disagreement, missingness, and entropy,
    with source validity from normalized out-of-fold AUROC);
  - `experiments/` — text/numeric baselines, the Qwen label trainer with strict
    participant-level OOF splitting, the matched-OOF comparison against a
    content-frozen reference ensemble, full-loop rollout collection
    (prepare → collect → merge → evaluate → training package → cross-fit),
    offline loop-aligned post-training (Action-SFT, Revision-SFT, ORPO with
    current-policy rollout refresh), and metric-only evaluation entry points.
- `scripts/` — command-line orchestration only; core algorithms live in
  `src/`. Every entry point is registered in `scripts/README.md`.
- `configs/` — experiment, task, and textualization configurations.
- `schema/` — JSON Schemas for evidence units and rethinking contracts.
- `examples/` — synthetic contract examples.

## Install

```bash
pip install -e .                # core (numpy / pandas / PyYAML)
pip install -e .[experiments]   # + scikit-learn / scipy baselines
pip install -e .[qwen-train]    # + torch / transformers / peft training stack
```

## Data

DAIC-WOZ and E-DAIC are distributed under signed data use agreements, and
D-Vlog under its own release terms. **This repository does not redistribute any
dataset**; only de-identified released features are processed and raw
audio/video is never handled. Point the scripts at your local dataset copies
and model weights through environment variables (`PROJECT_ROOT`, `MODEL_PATH`,
`DATA_ROOT`, dataset-specific label paths); each script documents its
variables in its header, and `configs/` carries the per-experiment values.

## Evaluation discipline

- The three datasets are fitted, trained, and model-selected independently:
  no merged training and no cross-dataset weight sharing.
- Reference statistics are fitted on training folds only. Held-out runs first
  freeze label-free blind predictions with SHA-256 manifests; a metric-only
  evaluator joins labels afterwards. Model selection, thresholds,
  calibration, and ensembling never touch held-out labels.
- Prompts and model-visible inputs never contain outcome-derived fields;
  supervision targets live in physically separate outcome files.

## License

MIT, see `LICENSE`.
