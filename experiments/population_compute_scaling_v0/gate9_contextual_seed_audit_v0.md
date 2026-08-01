# Gate-9 contextual training seed audit v0

## Status

**INDEPENDENT READ-ONLY AUDITOR FOR ONE COMPLETED CONTEXTUAL-TRAINING SEED.
NO TRAINING, RETRAINING, CHECKPOINT SELECTION, SCIENTIFIC TEST GENERATION,
POPULATION EXECUTION OR GATE-9 FINAL CLASSIFICATION IS ADMITTED.**

Qualified source execution head:

```text
bdc1af9bc65b94b01ae3946977686bd90158786f
```

Qualified architecture branch head:

```text
c689cc3f38f6f6f642916ee1a702d7de7bd0e43b
```

The auditor source originally transcribed that architecture identity as the
38-character value:

```text
c689cc3f38f6f642916ee1a702d7de7bd0e43b
```

The audit entrypoint corrects this only when the source still contains that
exact known typo. The replacement must be a 40-character lowercase Git SHA and
must equal `origin/agent/gate9-contextual-worker-architecture-v0`. Any different
source value, malformed replacement, or branch mismatch fails closed. The
source training artifact is never modified, and the correction is recorded in
the audit report and audit run configuration.

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
- exact branch, software, seed, protocol and corrected architecture identities;
- exact run-config field set, Python types and values;
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

The report contains source identities, the explicit architecture-identity
correction, reconstructed counts and metrics, checkpoint structure, seed
outcome and the still-closed scientific-test flag.

## Qualification

CI constructs a complete synthetic 32,768-row failed-seed artifact with the
same aggregate counts reported by seed 0. It proves exact metric
reconstruction, fixed-final checkpoint validation, manifest tamper rejection,
non-finite tensor rejection and absence of trainer/operator/scientific runtime
imports.

It also proves:

- cache-independent compilation of the checked-out auditor source;
- the known typo is exactly 38 lowercase hexadecimal characters;
- the corrected identity is exactly 40 lowercase hexadecimal characters;
- the corrected identity equals the remote architecture branch head;
- the real Windows run-config shape passes exact validation;
- truncated, extra-field and type-drift variants fail closed.

No CI artifact is evidence about the real seed-0 run.

## Failed audit-attempt preservation

The audit runner creates a new output root before the read-only source audit.
Three earlier tooling failures therefore produced three empty roots which must
remain preserved and must never be reused:

```text
F:\gate9_contextual_training_seed0_audit_v0
F:\gate9_contextual_training_seed0_audit_v0_retry1
F:\gate9_contextual_training_seed0_audit_v0_retry2
```

They contain no scientific audit report. The next qualified execution uses a
new `retry3` root.

## Next boundary

After the real seed-0 audit report is uploaded and independently checked, the
failure can be recorded as an immutable result-only slice. The preregistered
seed-1 and seed-2 executions may then proceed for replication, while Gate-9
scientific local and graph tests remain permanently closed under v0.
