# Gate-9 failure decomposition stage-1 execution v0

## Status

**SEED-SPECIFIC EXECUTION SURFACE FOR `single_operator_query_fit` ONLY. STAGES
2–4, GATE-9 V0 SCIENTIFIC WORLDS, POPULATION EXECUTION, DIAGNOSTIC
CLASSIFICATION, RETRAINING, AND CHECKPOINT SELECTION REMAIN CLOSED.**

Qualified diagnostic protocol head:

```text
8deca15aef78d8636b07570aff044f9b7ae31928
```

Each invocation executes exactly one ordered diagnostic seed index `0`, `1`, or
`2` under initialization seeds `910900`, `910901`, or `910902`.

## Stage-1 question

Can the frozen 19,649-parameter Gate-9 worker and optimizer fit one complete
non-support byte mapping when operator context is constant?

This is a prerequisite test only. Passing stage 1 does not establish causal
support use or unseen-operator induction.

## Frozen data

The single operator is the exact counter at `2^56`. Its public support contains
the frozen nine support inputs in the qualified global order. Training and
evaluation contain every one of the 247 remaining byte values exactly once.

```text
operators                    1
unique non-support queries   247
unique targets               operator-defined
public-support oracle        required exact
```

A deterministic SHA-256 binds the operator counter, operator key, support pairs,
queries, targets, and oracle targets.

## Frozen training

```text
steps                         1,024
batch size                    247
examples per step             complete fixed dataset
examples seen                 252,928
optimizer                     AdamW
maximum LR                    1e-3
minimum LR                    1e-4
warmup                        16 steps
schedule                      cosine after warmup
betas                         0.9, 0.95
epsilon                       1e-8
weight decay                  1e-4
gradient clip                 1.0
precision                     float32
AMP / TF32 / compile          off / off / off
checkpoint                    fixed final step 1,024 only
```

There is no random minibatch ordering, early stopping, best-checkpoint search,
seed replacement, or retry.

## Evaluation and pass rule

Evaluation uses the same complete 247-query mapping. It records full-context,
query-only, and public-support-oracle predictions per episode.

A seed passes stage 1 only when:

```text
exact byte accuracy    >= 0.995
bit accuracy           >= 0.999
oracle accuracy        == 1.0
```

Context deltas are not required because support is constant in this stage.

The execution slice records one seed result only. It does not combine the three
seeds or classify the diagnostic. A later result-only slice must independently
audit each artifact and apply the all-three-seeds stage rule.

## Immutable artifact root

Each successful invocation writes a new, otherwise unused output root:

```text
run-config.json
git-head.txt
git-status.txt
manifest.sha256
seed-N/train-steps.jsonl
seed-N/evaluation-per-episode.jsonl
seed-N/selected-checkpoint.pt
seed-N/summary.json
```

The training ledger contains exactly 1,024 rows. The evaluation ledger contains
exactly 247 rows. The recursive manifest binds every file. The checkpoint
contains the fixed-final state only: 17 finite float32 tensors and 19,649
parameters, with no optimizer state.

A failed pass threshold is a valid diagnostic result and does not trigger
retraining. An infrastructure failure preserves its incomplete output root and
cannot be overwritten silently.

## Closed boundaries

The runtime selects only protocol stage index `0`. It contains no paired
collision identities, held-in multi-operator range, unseen-operator range,
scientific assignment key, test-world generator, population runtime, or
diagnostic classifier.

Stages 2–4 cannot execute through this branch. Gate-9 v0 remains immutably
classified `G9_NOVEL_OPERATOR_INDUCTION_FAILED`.

## Qualification

CI installs the exact Torch/NumPy family and performs CPU contract checks only:

- exact stage/data/oracle materialization;
- exact learning-rate schedule;
- one forward/backward step;
- fixed checkpoint schema and tensor geometry;
- complete evaluation-ledger construction;
- ASCII Windows PowerShell wrapper smoke before imports or output creation;
- exact six-file execution scope;
- absence of later-stage and scientific execution surfaces.

CI does not run the 1,024-step diagnostic and emits no scientific evidence.
