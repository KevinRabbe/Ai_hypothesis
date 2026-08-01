# Gate-9D fast diagnostic harness v0

## Status

**DEVELOPMENT-ONLY FAILURE LOCALIZATION. NOT CONFIRMATION, NOT A NEW GATE-9
RESULT, AND NOT AUTHORITY TO OPEN LATER DIAGNOSTIC STAGES OR POPULATION
SCIENCE.**

The frozen Gate-9D stage-1 result remains unchanged. This harness replaces the
slow per-seed publication loop for small architecture diagnostics with one
local command, one aggregate directory, and one ZIP upload.

## Question

The frozen contextual worker failed to fit one fixed affine byte operator in
all three stage-1 seeds. This harness asks where that failure first appears:

1. **Optimizer/data wiring** — can a zero-initialized byte lookup move every
   observed output bit to the correct side of zero?
2. **Affine representation** — can a linear head over all 256 Walsh parity
   features fit the GF(2) affine mapping?
3. **Current query path** — can the frozen worker fit the mapping when its
   support summary is fixed to zero?
4. **Current full path** — can the frozen worker fit the same mapping with the
   constant nine-pair support context present?

A plain real-valued linear layer over eight input bits is deliberately not used:
GF(2) parity is not generally linearly separable in that representation.

## Frozen comparison

Every variant uses:

- the same 247 non-support byte queries and exact targets;
- the same three initialization seeds `910900`, `910901`, and `910902`;
- 1,024 full-batch AdamW steps;
- the frozen warmup-plus-cosine learning-rate schedule;
- fixed final-step evaluation;
- exact thresholds `0.995` byte accuracy and `0.999` bit accuracy.

The current query-only and full-context variants are constructed from the same
initial weights within each seed by resetting the deterministic seed before
model construction.

## Recorded diagnostics

The harness records checkpoints at steps:

```text
0, 1, 16, 64, 128, 256, 512, 1024
```

Each checkpoint contains loss, exact and bit accuracy, gradient norm, active
gradient element count, parameter norm, and parameter-update norm.

The final bundle contains:

```text
aggregate-summary.json
curves.jsonl
final-runs.jsonl
predictions.jsonl
git-head.txt
git-status.txt
run-config.json
manifest.sha256
```

The runner also creates one adjacent ZIP containing the complete directory.
No model checkpoint is written by default; this keeps the development audit
small and prevents checkpoint-selection behavior.

## Ordered development diagnoses

The aggregate diagnosis is fail-closed and ordered:

```text
G9D_FAST_LOOKUP_PIPELINE_FAILED
G9D_FAST_LOOKUP_PIPELINE_MIXED
G9D_FAST_PARITY_REPRESENTATION_FAILED
G9D_FAST_PARITY_REPRESENTATION_MIXED
G9D_FAST_CURRENT_QUERY_PATH_FAILED
G9D_FAST_SUPPORT_CONTEXT_RESCUES_QUERY_PATH
G9D_FAST_SUPPORT_PATH_INTERFERENCE
G9D_FAST_SUPPORT_PATH_MIXED
G9D_FAST_CURRENT_WORKER_MIXED
G9D_FAST_STAGE1_FAILURE_NOT_REPRODUCED
```

These labels are development evidence only. They do not replace the frozen
classification `G9D_BASIC_QUERY_MAPPING_FAILED` and cannot mutate any prior
result record.

## Closed boundaries

The harness contains no:

- stage-2, stage-3, or stage-4 execution path;
- Gate-9 local or graph scientific-world generator;
- population runtime;
- early stopping or checkpoint selection;
- confirmation or immutable-result publisher;
- mutation path for any frozen Gate-9 or Gate-9D result.
