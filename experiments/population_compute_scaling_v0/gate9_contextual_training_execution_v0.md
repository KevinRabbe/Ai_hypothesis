# Gate-9 contextual worker training execution v0

## Status

**SEED-SPECIFIC EXECUTION SURFACE FOR THE FROZEN TRAINING PROTOCOL — LOCAL
AND GRAPH SCIENTIFIC OPERATORS, THE SCIENTIFIC ASSIGNMENT KEY, TEST WORLDS,
POPULATION EXECUTION AND GATE-9 RESULT CLASSIFICATION REMAIN CLOSED.**

Frozen training-protocol head:

```text
1228c19cbf85da4ab738c3355c58f946cd6a965c
```

Each invocation trains exactly one seed (`0`, `1`, or `2`) and writes one
independent immutable checkpoint/validation evidence root. The three runs are
never pooled or selected in this phase.

## Exact run

One seed executes:

```text
262,144 unique training operators
512 episodes per batch
512 optimizer steps
one novel query per operator
fixed-final step-512 checkpoint
32,768 disjoint validation operators
64 validation batches
```

Training and validation order, queries, optimizer, learning-rate schedule,
precision, determinism and admission thresholds come only from the qualified
protocol.

## Episode materialization

The data layer reconstructs each affine operator directly from the exact
SplitMix64 key and unit-triangular factors. It emits only:

```text
nine qualified support input bytes
nine support output bytes
one novel query byte
one answer byte
```

Contract regressions compare the fast path with the exact public-support oracle.
Training counters stay below the validation range. Validation counters stay
within `[2^32, 2^32 + 32768)`.

Runtime-observed Boolean coverage vectors reject any repeated or missing
training or validation operator ordinal. Summaries use the observed counts, not
only protocol constants.

## Validation controls

The validation support shuffle is one SHA-256-derived nonzero rotation of the
complete 32,768-episode validation sequence. It is a full derangement and
preserves every support set exactly once.

Every validation row records:

```text
operator ordinal and counter
query and answer
full prediction
shuffled-context source and prediction
query-only prediction
oracle prediction
all four correctness flags
```

The query-only path supplies no support rows. The oracle is reconstructed from
public support and must match the direct private target.

## Checkpoint and evidence

The selected checkpoint contains only the fixed step-512 model state and frozen
metadata. It excludes optimizer state. The model state must contain exactly 17
finite float32 tensors and exactly 19,649 parameters.

Each output root contains:

```text
run-config.json
git-head.txt
git-status.txt
manifest.sha256
seed-N/train-steps.jsonl
seed-N/validation-per-episode.jsonl
seed-N/selected-checkpoint.pt
seed-N/summary.json
```

The training ledger has exactly 512 rows. The validation ledger has exactly
32,768 rows. The recursive manifest binds all files with relative POSIX paths.

The summary includes:

- observed unique training/validation coverage;
- final loss and immutable checkpoint SHA-256;
- full byte and bit accuracy;
- shuffled-context, query-only and oracle accuracy;
- whether that seed passes the preregistered admission rule;
- dataset/evidence SHA-256 identities;
- execution timing and device;
- explicit closed-boundary flags.

## Infrastructure boundary

A validation failure is a valid result and does not trigger retraining. An
infrastructure failure preserves its incomplete output root; it must not be
silently overwritten or confused with a completed seed artifact.

The PowerShell wrapper requires:

```text
Python 3.11.9
Torch 2.9.1+cu130
NumPy 2.3.5
CUDA
clean exact execution branch
new output directory
```

It enables deterministic cuBLAS workspace configuration and rejects software,
branch, tree or output-root drift before training.

## Qualification

CI does not run the frozen 262,144-episode training. It performs only:

- exact module compilation;
- fast-material/public-oracle equivalence checks;
- representative frozen batch and range checks;
- complete validation derangement proof;
- one contract-range CPU optimizer smoke step;
- exact state/checkpoint-schema checks;
- recursive-manifest checks;
- Windows wrapper smoke before imports, CUDA, output or optimizer creation;
- exact seven-file branch-scope proof.

No CI artifact is scientific evidence.

## Closed boundary

This slice opens training and validation only when the user runs the qualified
wrapper from the exact branch. It contains no local-test operator generation,
graph-test operator generation, scientific assignment key, test-world builder,
population runtime, bootstrap classifier or Gate-9 final outcome.

## Next boundary

After all three seed roots complete, each seed must be independently audited and
recorded. Only then may a separate all-seed checkpoint-admission finalizer apply
the frozen classifier. If any seed fails validation, Gate-9 scientific test
generation remains closed.
