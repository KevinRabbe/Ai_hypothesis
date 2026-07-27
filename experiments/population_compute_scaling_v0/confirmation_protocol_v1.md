# Population Compute Gate v0 — relay v1 frozen confirmation protocol

## Status

**Frozen before confirmation is opened.**

This document specifies the exact confirmation execution for the canonical repaired relay-v1 protocol. It is intentionally narrower than the broader research roadmap.

The clean seed-0 development result and the serial schedule equivalence result are already recorded separately. Neither may be reinterpreted after confirmation.

## Scientific claim being tested

The confirmation question is:

> With one learned parameter set per training seed and the learned parameter count held fixed across runtime population sizes, does the canonical repaired relay-v1 system reproducibly convert additional runtime population computation and additional available distributed source scope into increasing end-to-end capability?

The confirmation result does **not** test whether parallel population state is more capable than equal-work serial execution. The serial-control result already established that the current repaired relay computation is schedule-equivalent at matched learned worker-update count.

## Canonical implementation

Confirmation must use only:

- experiment: `population-compute-relay-training-v1`;
- protocol: `relay-protocol-v1-normalized-gate-supervised`;
- benchmark: `collective-relay-v1-answer-frontier`;
- communication condition: `sparse_shared_v1`;
- matched control: `no_communication`;
- canonical CLI: `ai_hypothesis.population_compute.run_relay_scaling_v1`.

No diagnostic script is admissible for the final confirmation result.

## Frozen training seeds

Use exactly three new independent training seeds:

```text
1
2
3
```

Training seed 0 is reserved as the development/canonical-reproduction seed and is not counted toward confirmation reproducibility.

Do not add extra seeds after results are visible in an attempt to rescue a failed seed. Additional seeds may be studied later only as a separately declared replication experiment.

## Frozen learned architecture and training

For every confirmation seed:

```text
steps                    = 2000
train_batch_size         = 64
learning_rate            = 0.0003
weight_decay             = 0.0001
gradient_clip_norm       = 1.0
gate_supervision_weight  = 1.0
state_width              = 64
message_width            = 24
```

Training rotates deterministically through relay-2, relay-4 and relay-8 and through their admissible first-complete population thresholds using the already-versioned training seed schedule.

The gate-supervision target is training-only. No oracle chain identity is available to inference.

Each trained checkpoint must be persisted and freshly reloaded before confirmation evaluation.

## Frozen evaluation

For each of seeds 1, 2 and 3:

```text
evaluation_split       = confirmation
evaluation_world_count = 1000 per relay difficulty
evaluation_batch_size  = 64
population_sizes       = 1 / 4 / 16 / 64 / 256
execution_mode         = eager
```

Every seed must evaluate the same frozen confirmation worlds.

The communicating and no-communication conditions at each population point must use exactly the same worlds and therefore exactly the same information-complete counts.

## Hardware/runtime control

All three seeds must execute sequentially inside one GitHub Actions job on one runner instance.

The workflow must record at least:

- runner OS/image metadata available from Actions;
- Python version;
- Torch version;
- CPU identity from `lscpu`;
- execution mode (`eager`);
- per-seed checkpoint fingerprints.

Running all seeds in one job prevents seed-to-seed hardware changes from being silently mixed into the confirmation comparison.

## Frozen per-seed Gate-v0 rule

For each training seed independently:

1. at least **two relay difficulty tiers** must each satisfy both:
   - 256-worker raw solve rate exceeds the 1-worker raw solve rate by at least **5 percentage points**;
   - at least **three of four** adjacent population steps are non-decreasing within a **1 percentage point** tolerance;
2. at least **one relay difficulty tier** must have a 256-worker communicating endpoint at least **5 percentage points** above its matched no-communication endpoint;
3. learned parameter count and exact parameter fingerprint must remain identical across every point in that seed's compared curve;
4. communicating and no-communication conditions must have identical benchmark scope/information-complete counts;
5. worker-update, communication and scope accounting must remain valid.

Conditional solve-given-complete and solve-given-incomplete are reported but have no post-hoc pass threshold.

## Frozen cross-seed aggregation

The final relay-v1 confirmation gate passes only if **all three seeds 1, 2 and 3 independently pass the per-seed rule above**.

There is no majority rule.

```text
3 / 3 pass -> confirmation Gate-v0 positive
anything else -> confirmation Gate-v0 negative for this protocol
```

This conservative aggregation is frozen before confirmation results are visible.

## Serial-control interpretation

The already-qualified serial schedule is mathematically equivalent to the repaired normalized parallel relay at matched source scope and learned worker-update count.

Therefore even a positive confirmation result supports:

> fixed learned parameters + additional reusable runtime neural computation/source scope can reproducibly produce additional capability.

It does **not** by itself support:

> simultaneous wide population state provides more capability per learned update than an equal-work serial schedule.

Parallel width may still provide a latency/throughput advantage; that is a later resource-frontier experiment.

## Stop rule

After the confirmation workflow begins:

- do not change benchmark generation;
- do not change training hyperparameters;
- do not change gate-supervision weight;
- do not change communication normalization;
- do not change population sizes;
- do not change confirmation world count;
- do not add seeds to rescue the result;
- do not reinterpret the frozen pass rule.

Any required scientific change creates a new protocol/version and invalidates the current confirmation attempt rather than silently modifying it.
