# Gate-9D affine feature bridge v0

## Status

**DEVELOPMENT-ONLY REPRESENTATION REPAIR. NOT A NEW GATE-9 RESULT, NOT A
CLASSIFICATION OF THE FROZEN GATE-9D LADDER, AND NOT AUTHORITY TO OPEN
POPULATION SCIENCE.**

The query-capacity diagnostic established that the original worker's query path
has a severe raw-bit parity mismatch:

```text
current query-only, 1,024 steps     70–90 / 247
current query-only, 4,096 steps     126–142 / 247
raw-bit tanh 32, 1,024 steps        13–17 / 247
raw-bit tanh 64, 1,024 steps        27–39 / 247
Walsh tanh 24, 1,024 steps          247 / 247 in all seeds
```

The remaining question is whether a representation-aware worker can use public
support context causally and generalize to disjoint unseen operators without
operator keys, per-operator parameters, or a global operator lookup.

## Public affine feature bridge

Gate-9 public support contains exactly:

```text
f(0), f(e0), f(e1), ..., f(e7)
```

For each output bit, these values publicly determine:

```text
bias = f(0)
mask_i = f(e_i) XOR bias
```

The bridge performs only that local public-support feature transformation. It
does not reconstruct or expose the 64-bit operator key, SplitMix counter,
triangular factors, world identity, worker identity, or any per-operator state.

For the incoming query it computes one parity sign and one bias sign per output
bit. A single learned decoder shared across every output bit and every operator
must compose those two signs into the output logit.

## Learned machinery

```text
2 signed features
-> Linear(2, 16)
-> tanh
-> Linear(16, 1)
```

The exact learned parameter count is:

```text
48 hidden-layer parameters
17 output-layer parameters
65 total learned parameters
```

This is approximately 302.29 times smaller than the failed 19,649-parameter
Gate-9 v0 worker. The reduction is intentional: this diagnostic asks whether
the failure was representational, not whether additional capacity can hide it.

## Fresh disjoint operators

This diagnostic does not reuse the frozen Gate-9D stage-4 identities.

```text
training operators     256 counters from 2^57
validation operators    64 counters from 2^57 + 0x1000
queries per operator    247 non-support bytes
training examples       63,232
validation examples     15,808
```

The ranges are disjoint from each other, Gate-9 v0, and all frozen Gate-9D
failure-decomposition ranges.

## Training

Each of the three frozen diagnostic initialization seeds trains independently:

```text
AdamW
512 steps
batch size 512
16-step linear warmup
cosine decay to 1e-4
fixed final step 512
CPU float32 deterministic execution
no early stopping
no checkpoint selection
no retries
```

No model checkpoint is written. The complete run is fast enough to preserve one
small evidence bundle.

## Required controls

The fixed feature bridge itself must reproduce every validation answer exactly.
This proves only representation correctness and is recorded as an oracle
control.

The learned decoder is evaluated under:

1. correct public support;
2. a one-operator rotation of validation support, which is a complete
   derangement;
3. query-only features with no support information;
4. the fixed bridge oracle.

A seed passes only when:

```text
full exact accuracy         >= 0.995
full bit accuracy           >= 0.999
oracle exact accuracy       == 1.0
full - shuffled exact       > 0.50
full - query-only exact     > 0.50
```

All three seeds must pass for:

```text
G9D_AFFINE_FEATURE_BRIDGE_PASSES
```

Mixed seeds produce `G9D_AFFINE_FEATURE_BRIDGE_MIXED`; three failures produce
`G9D_AFFINE_FEATURE_BRIDGE_FAILED`.

## Evidence bundle

One command writes:

```text
aggregate-summary.json
curves.jsonl
final-runs.jsonl
evaluation.jsonl
git-head.txt
git-status.txt
run-config.json
manifest.sha256
<output-root>.zip
```

The evaluation ledger records full, shuffled, query-only, and oracle
predictions for every validation episode and every seed.

## Interpretation boundary

A positive result would establish that the public supports contain sufficient
causal information and that a tiny shared learned decoder generalizes once the
operator family is represented in its natural affine/Walsh coordinates.

It would not yet establish population capability scaling. It would justify a
new contextual worker architecture and a fresh qualified Gate-9 execution
stack. It would not retroactively alter the frozen negative Gate-9 v0 result.
