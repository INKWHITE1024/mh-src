# Two-pass workflow examples

The two JSON files in this directory are contract examples, not predictions and not
participant annotations. `initial_assessment.example.json` is accepted by the first-pass
schema. `revision_assessment.example.json` cites only atomic IDs available in that retrieval.

The task definition is separate from the current sample:

```text
configs/tasks/phq8_depression_binary.json
```

Prepare an exact first-pass Qwen chat request:

```bash
rethink-workflow prepare-initial \
  --session-dir artifacts/local_examples/native/daic_woz/300 \
  --task configs/tasks/phq8_depression_binary.json \
  --output outputs/rethinking/daic_300.initial_messages.json
```

Validate an initial completion, apply the trigger, and render targeted evidence:

```bash
rethink-workflow prepare-revision \
  --session-dir artifacts/local_examples/native/daic_woz/300 \
  --task configs/tasks/phq8_depression_binary.json \
  --policy-config configs/rethinking/heuristic_baseline.json \
  --initial-output examples/rethinking/initial_assessment.example.json \
  --output-dir outputs/rethinking/daic_300_revision
```

The default second pass retrieves at most three segments, four time-spanning atomic
windows per segment, and three observations per modality. Canonical IDs remain in the
retrieval manifest for auditing but are not exposed to the model text.

## Query-Grounded Atomic Retrieval

`evidence_query.example.json` and `atomic_selection.example.json` are synthetic
contract examples for the query-grounded atomic retrieval path. The E IDs illustrate
grounding syntax only and are not tied to a participant or result.

The query-guided path first exposes a label-free Segment retrieval map, then asks the
model for an ordinary-English fixed-slot query. A deterministic retriever creates
low-detail candidate cards, and the model may select only E IDs present in those
cards. Full atomic measurements are released only after validation.

