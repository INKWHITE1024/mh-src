# Configuration map

| Directory | Contents | Rule |
|---|---|---|
| `textualization/` | Dataset adapters, windows, sources, and native Evidence protocol parameters | Refit references whenever reference-digest inputs change |
| `tasks/` | Task definitions and thresholds (depression, PTSD) | Label rules are frozen before training |
| `rethinking/` | Trigger, retrieval, and revision policy configs | Must never read labels implicitly |
| `experiments/` | Training, OOF, D-Vlog, and full-loop experiment configs | Config names must match the output manifests |

Dataset keys are `daic_woz`, `e_daic`, and `d_vlog`. `native` is the only
Evidence protocol version.
