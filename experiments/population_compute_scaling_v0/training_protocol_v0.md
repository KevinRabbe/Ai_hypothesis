# Collective Relay Training Protocol v0

## Status

This document freezes the first train-once protocol used to make the fixed-parameter population-compute development curve runnable.

It was defined before any trained development curve was interpreted.

It does **not** open confirmation and it does not define a new success threshold. Gate-v0 interpretation remains owned by the parent experiment contract.

## Purpose

The development curve asks whether one fixed learned parameter set becomes more capable when more runtime population states and communication are available.

A training protocol can accidentally manufacture such a curve if it trains only at the largest population size. Smaller populations would then be out-of-distribution and lower capability could reflect training exposure rather than useful population computation.

Training v0 therefore makes population exposure an explicit controlled variable.

## One learned model

One `RelayPopulationModel` is initialized for one training seed and trained once.

The resulting checkpoint is the only learned model used by every development condition:

- population 1;
- population 4;
- population 16;
- population 64;
- population 256;
- `no_communication`;
- `sparse_shared_v0`.

Evaluation may change runtime state count and communication mode. It may not change trainable parameters.

The checkpoint is saved, loaded into a fresh model instance, and rejected if either exact trainable-parameter count or SHA-256 parameter fingerprint changes.

## Population-balanced supervised exposure

The supervised training population ladder is:

```text
4, 16, 64, 256
```

Training steps cycle through those four sizes in that order. Therefore each size receives the same number of training batches when the step count is divisible by four, and differs by at most one batch otherwise.

Population 1 is deliberately excluded from supervised relay training. Every benchmark difficulty requires at least two distinct chain records, so one local state can never contain complete relay information. Training an exact target on an information-incomplete one-worker input would reward guessing rather than relay computation.

## Complete-information training only

Every supervised training batch satisfies:

```text
world.scope_threshold <= active_workers
```

Therefore every training world contains the complete required relay chain inside the active prefix.

This prevents the optimizer from being asked to predict an answer that is not determined by its available information.

The controlled scope threshold itself is benchmark metadata and is never passed to the neural model.

## Difficulty and scope schedule

For each active population size:

1. cycle through relay difficulties that can physically fit in that population;
2. within each difficulty, cycle through the controlled scope thresholds that are complete at that population;
3. use deterministic threshold-matched world seeds from the training-only seed range.

Examples:

- population 4 can train relay-2 / relay-4 worlds whose threshold is 4;
- population 16 can train relay-2 / relay-4 / relay-8 and thresholds no larger than 16;
- population 64 includes lower-threshold worlds plus threshold-64 worlds, so it learns with additional distractor records rather than seeing only first-complete inputs;
- population 256 likewise sees all admissible scope thresholds.

This matters because a larger active population must work both when the final required fact appears near its frontier and when the complete chain was already available at a smaller prefix.

## Training communication

Training uses only:

```text
sparse_shared_v0
```

The `no_communication` condition is an ablation of the **same trained checkpoint**, not a separately optimized model. This tests whether recurrent population-produced communication contributes beyond scope plus the final pooled readout.

No second communication design is introduced by this protocol.

## Recurrent depth

For one world, recurrent rounds equal the benchmark hop count:

```text
relay-2 -> 2 rounds
relay-4 -> 4 rounds
relay-8 -> 8 rounds
```

Recurrent depth is therefore tied to the amount of sequential relay work required by the benchmark rather than silently increased with population size.

## Loss and optimizer defaults

Neural node identities use the frozen 12-bit encoding.

Training target: final answer-key bits.

Loss:

```text
binary cross entropy with logits
```

Default optimizer/configuration:

```text
AdamW
learning_rate = 3e-4
weight_decay = 1e-4
gradient_clip_norm = 1.0
steps = 2000
batch_size = 64
state_width = 64
message_width = 24
```

These are development settings, not claims of optimality.

Ordinary training debugging may change them before a viable development configuration is frozen. Such changes must remain explicit in checkpoint/result provenance. Gate criteria, benchmark scope rules and parameter-identity rules do not change with training tuning.

## Seed isolation

Training and evaluation use disjoint reserved deterministic seed spaces:

```text
training:     starts at 10,000,000
development:  starts at 1,000,000,000
confirmation: starts at 2,000,000,000
```

Each user-visible training/benchmark seed owns a bounded block. Validation rejects a seed index that would cross into the next split.

Within a training step, a bounded disjoint seed segment is searched deterministically for worlds with the required controlled scope threshold.

## Development evaluation

After checkpoint reload, the exact same weights are evaluated on the same development worlds at:

```text
1, 4, 16, 64, 256
```

for both:

```text
no_communication
sparse_shared_v0
```

Each condition records the canonical scope/capability decomposition:

- task count;
- solved count / raw solve rate;
- information-complete count / rate;
- solved count / rate given complete information;
- solved count / rate given incomplete information;
- worker updates;
- communication volume;
- state-memory estimate;
- wall time;
- exact learned parameter count and fingerprint.

Communication and no-communication conditions at the same population point must expose identical information-complete counts.

## Interpretation

The development result must be read as:

```text
population size
    -> scope availability
    -> utilization of available distributed information
    -> raw capability
```

A rising raw solve curve alone is insufficient evidence for population computation.

Particularly important cases:

- raw capability rises, conditional solve stays weak -> mostly scope exposure;
- conditional solve rises with population -> stronger system-level computation signal;
- `sparse_shared_v0` beats `no_communication` on matched scope -> evidence that recurrent communication contributes;
- all curves stay flat -> negative result or training failure to diagnose before any architecture expansion.

## Confirmation lock

The development CLI has no confirmation option.

Confirmation remains unopened until:

1. training mechanics are demonstrably functional;
2. one development training/model configuration is frozen;
3. the existing Gate-v0 criteria remain unchanged;
4. at least three independent training seeds are prepared for the confirmation protocol.

Do not add a confirmation flag as a convenience during development.
