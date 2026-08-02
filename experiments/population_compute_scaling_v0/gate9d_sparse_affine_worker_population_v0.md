# Gate-9D sparse affine worker population v0

## Status

**DEVELOPMENT-ONLY MECHANISTIC POPULATION EXECUTION OF THE VERIFIED AFFINE
BRIDGE. THIS DOES NOT CLAIM AUTOMATIC COORDINATE DISCOVERY, DOES NOT MODIFY ANY
FROZEN RESULT, AND DOES NOT OPEN LATER GATE-9D OR POPULATION-SCIENCE STAGES.**

The affine bridge established that public support inputs `0 + {e_i}` provide an
exact coordinate system for the frozen affine operator family. A separate raw
shared-worker prototype remained at chance, so the next question is narrower:

> Can the exact bridge be organized as sparse population computation whose
> communication cost is independent of nominal population size?

## Population rule

Every worker owns one observed support pair `(x, f(x))` and receives the query.
All workers reuse the same fixed local rule.

### Round 1 — bias broadcast

Only the worker with `x = 0` emits the eight-bit bias `f(0)`.

### Round 2 — sparse basis contributions

A worker emits only when:

```text
x is one-hot
and
the corresponding query bit is active
```

Its message is:

```text
f(x) XOR f(0)
```

The organism XOR-reduces active messages and then XORs the bias broadcast.
Workers with nonzero non-one-hot inputs are distractors and never emit.

## Population axis

```text
9
16
64
256
```

The first nine workers are the genuine public support interactions. Additional
workers receive deterministic irrelevant support pairs. The population rule is
unchanged and has zero learned parameters at every size.

Useful messages per episode are bounded by:

```text
1 bias message + popcount(query) contribution messages
```

They do not grow with nominal population size.

## Controls

- deterministic permutation of all workers;
- support outputs shifted across operators;
- no round-1 bias broadcast.

The full and permuted populations must be exact at every population size. Both
causal controls must remain below two percent exact accuracy.

## Fresh evidence range

```text
operator start  2^57 + 0x2000
operator count  128
queries         247 per operator
episodes        31,616
```

This range is disjoint from the affine-bridge training and evaluation ranges.

## Interpretation boundary

A pass would establish that the verified affine representation can be executed
as sparse, permutation-invariant population computation with a fixed zero-
parameter rule and bounded communication.

It would not establish that generic workers discover the representation from
raw observations. Learned routing and coordinate discovery remain separate
future questions.

## Output

One command writes one directory and one ZIP containing:

```text
aggregate-summary.json
population-rows.jsonl
episodes.jsonl
git-head.txt
git-status.txt
run-config.json
manifest.sha256
```

No model checkpoint is produced.
