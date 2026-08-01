# Gate-9 contextual training seed audit v0

## Status

**INDEPENDENT READ-ONLY AUDITOR FOR ONE COMPLETED CONTEXTUAL-TRAINING SEED.
NO TRAINING, RETRAINING, CHECKPOINT SELECTION, SCIENTIFIC TEST GENERATION,
POPULATION EXECUTION OR GATE-9 FINAL CLASSIFICATION IS ADMITTED.**

Qualified source execution head:

```text
bdc1af9bc65b94b01ae3946977686bd90158786f
```

The first completed seed-0 run reported:

```text
final loss                 0.69307345
full exact accuracy        109 / 32768 = 0.003326416015625
full bit accuracy          131200 / 262144 = 0.50048828125
shuffled-context accuracy  112 / 32768 = 0.00341796875
query-only accuracy        117 / 32768 = 0.003570556640625
oracle accuracy            32768 / 32768 = 1.0
```

These values indicate chance-level output and no positive contextual dependence,
but the terminal summary alone is not the immutable result. This slice audits
the complete artifact before any subsequent seed is opened.

## Read-only audit

The auditor requires the original artifact root and a separate new audit-output
root. It never writes inside the source artifact.

It binds the four terminal identities:

```text
summary
validation ledger
selected checkpoint
source manifest
```

Then it independently verifies:

- exact source manifest ordering, file set and every recursive SHA-256;
- clean execution Git status and exact execution head;
- exact branch, software, seed, protocol and architecture identities;
- 512 contiguous training rows and 262,144 reported episodes;
- the frozen linear-warmup/cosine learning rate at every step;
- finite losses, gradient norms and wall times;
- unique batch-allocation and query hashes;
- 32,768 contiguous validation rows;
- complete unique validation-operator coverage;
- exact counter/ordinal identity inside the frozen validation interval;
- novel queries outside the nine public support inputs;
- prediction ranges and all recorded correctness flags;
- full, bit, shuffled-context, query-only and oracle metrics;
- exact oracle accuracy of `1.0`;
- safe `weights_only=True` checkpoint loading;
- exact fixed-final checkpoint metadata;
- all 17 tensor names, shapes, float32 dtypes and finiteness;
- exactly 19,649 checkpoint parameters;
- exact reconstruction of the preregistered seed-admission rule.

The auditor does not import the trainer, training-data materializer, worker
architecture, operator generator or graph-world runtime.

## Outcome semantics

One failed seed produces:

```text
G9_CONTEXTUAL_SEED_CHECKPOINT_ADMISSION_FAILED
```

Because the frozen Gate-9 checkpoint-admission rule requires all three ordered
seeds to pass, a validated failure also establishes:

```text
all_seed_admission_still_possible = false
scientific_test_generation_allowed = false
```

This does not authorize a hyperparameter change, extra training, earlier
checkpoint selection, seed replacement or retry. Seeds 1 and 2 remain useful
for preregistered replication and diagnosis, but cannot reverse the admission
outcome once seed 0 is independently confirmed.

## Audit output

The audit runner writes only to a new output root:

```text
gate9-contextual-seed-audit.json
run-config.json
git-head.txt
git-status.txt
manifest.sha256
```

The report contains source identities, reconstructed counts and metrics,
checkpoint structure, seed outcome and the still-closed scientific-test flag.

## Qualification

CI constructs a complete synthetic 32,768-row failed-seed artifact with the
same aggregate counts reported by seed 0. It proves exact metric
reconstruction, fixed-final checkpoint validation, manifest tamper rejection,
non-finite tensor rejection and absence of trainer/operator/scientific runtime
imports.

No CI artifact is evidence about the real seed-0 run.

## Next boundary

After the real seed-0 audit report is uploaded and independently checked, the
failure can be recorded as an immutable result-only slice. The preregistered
seed-1 and seed-2 executions may then proceed for replication, while Gate-9
scientific local and graph tests remain permanently closed under v0.
