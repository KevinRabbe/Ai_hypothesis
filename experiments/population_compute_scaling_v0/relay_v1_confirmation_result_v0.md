# Relay v1 frozen confirmation result v0

## Status

**Gate v0 POSITIVE — 3 / 3 independent confirmation training seeds passed the rule frozen before the confirmation split was opened.**

This result is the first untouched multi-seed confirmation for the canonical repaired relay-v1 protocol. It must be interpreted together with the already-qualified serial schedule equivalence result: the relay computation is serializable at matched learned worker-update count, so this result establishes reproducible **fixed-parameter runtime-compute scaling**, not an intrinsic function-level advantage from simultaneous wide state.

## Immutable provenance

- GitHub Actions run: `30239005530`
- artifact: `8642866100`
- experiment commit: `a2dcc322612dbdeb47ef3fd88243c01e33e9012b`
- experiment: `population-compute-relay-training-v1`
- protocol: `relay-protocol-v1-normalized-gate-supervised`
- benchmark: `collective-relay-v1-answer-frontier`
- confirmation seeds: `1 / 2 / 3`
- learned parameters per checkpoint: **26,669**
- population ladder: `1 / 4 / 16 / 64 / 256`
- evaluation worlds: `1,000` per relay difficulty
- execution mode: eager CPU
- Python: `3.11.15`
- Torch: `2.13.0+cpu`
- runner CPU: AMD EPYC 7763, 4 logical CPUs exposed by the runner
- artifact SHA-256: `4facdd7fc087202f3a54d26603f3baddd5983e6ed99bc85b1aa0d86c45ba4d0c`

## Frozen Gate rule

Before confirmation was opened, the protocol required every one of the three new training seeds to pass independently:

1. at least two relay tiers must each have >=5 percentage-point endpoint gain from 1 to 256 workers **and** at least 3/4 adjacent population steps non-decreasing within a 1-point tolerance;
2. at least one relay tier must have >=5-point communication endpoint advantage over matched no-communication;
3. exact parameter identity within each seed, matched benchmark scope, and valid accounting;
4. final Gate passes only if **3 / 3** seeds pass.

No majority rule or rescue seeds were permitted after results became visible.

## Training checkpoints

| Seed | Fingerprint | Initial loss | Final loss | Mean last 50 | Final relay | Final gate |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `c227ade9006e47bec17a2a3d5aedf6ac95a6a94607b96b9f52ab759905536c12` | 2.123916 | 0.080078 | 0.096852 | 0.058549 | 0.021529 |
| 2 | `14794971eb2702b22edbcf47d514577c0a5e00220559fb0a6bff50f26d4da364` | 2.118004 | 0.060768 | 0.077568 | 0.045992 | 0.014776 |
| 3 | `a9389c6fdd34f7a4d36796b0f67267153c6ed5a5ec05335abed63dc73825a479` | 2.123169 | 0.058486 | 0.068179 | 0.042465 | 0.016022 |

Each seed produced a different independently trained checkpoint fingerprint, while every population/control point inside that seed reused exactly the same checkpoint and the same 26,669 learned parameters.

## Raw communicating exact-solve curves

### Seed 1

| Active workers | relay-2 | relay-4 | relay-8 |
|---:|---:|---:|---:|
| 1 | 0.00% | 0.00% | 0.00% |
| 4 | 24.90% | 25.00% | 0.00% |
| 16 | 49.80% | 49.90% | 33.20% |
| 64 | 74.50% | 75.00% | 66.50% |
| 256 | 99.00% | 98.70% | 96.70% |

### Seed 2

| Active workers | relay-2 | relay-4 | relay-8 |
|---:|---:|---:|---:|
| 1 | 0.00% | 0.00% | 0.00% |
| 4 | 25.00% | 25.00% | 0.00% |
| 16 | 50.00% | 50.00% | 33.40% |
| 64 | 75.00% | 74.90% | 66.60% |
| 256 | 99.90% | 99.40% | 98.50% |

### Seed 3

| Active workers | relay-2 | relay-4 | relay-8 |
|---:|---:|---:|---:|
| 1 | 0.00% | 0.00% | 0.00% |
| 4 | 25.00% | 25.00% | 0.00% |
| 16 | 50.00% | 49.90% | 33.40% |
| 64 | 75.00% | 74.90% | 66.70% |
| 256 | 100.00% | 99.90% | 99.60% |

## Across-seed descriptive mean

| Active workers | relay-2 | relay-4 | relay-8 |
|---:|---:|---:|---:|
| 1 | 0.00% (0.0–0.0%) | 0.00% (0.0–0.0%) | 0.00% (0.0–0.0%) |
| 4 | 24.97% (24.9–25.0%) | 25.00% (25.0–25.0%) | 0.00% (0.0–0.0%) |
| 16 | 49.93% (49.8–50.0%) | 49.93% (49.9–50.0%) | 33.33% (33.2–33.4%) |
| 64 | 74.83% (74.5–75.0%) | 74.93% (74.9–75.0%) | 66.60% (66.5–66.7%) |
| 256 | 99.63% (99.0–100.0%) | 99.33% (98.7–99.9%) | 98.27% (96.7–99.6%) |

The raw curve closely tracks the benchmark's designed source-availability ladder rather than exposing an early-answer shortcut.

## Scope availability is unchanged by training seed

| Active workers | relay-2 | relay-4 | relay-8 |
|---:|---:|---:|---:|
| 1 | 0% | 0% | 0% |
| 4 | 25% | 25% | 0% |
| 16 | 50% | 50% | 33.4% |
| 64 | 75% | 75% | 66.7% |
| 256 | 100% | 100% | 100% |

### Critical shortcut check

**Exact solve given incomplete information was 0% for every seed, relay depth, and population point where incomplete worlds existed.**

This confirms that the relay-v1 answer-frontier repair remained effective on untouched confirmation worlds: the model did not solve worlds for which the required relay chain was absent from active scope.

## Capability given complete information

Across the three confirmation seeds, the range of exact solve rate conditional on all required chain information being available was:

| Active workers | relay-2 | relay-4 | relay-8 |
|---:|---:|---:|---:|
| 1 | — | — | — |
| 4 | 99.60–100.00% | 100.00–100.00% | — |
| 16 | 99.60–100.00% | 99.80–100.00% | 99.40–100.00% |
| 64 | 99.33–100.00% | 99.87–100.00% | 99.70–100.00% |
| 256 | 99.00–100.00% | 98.70–99.90% | 96.70–99.60% |

The fixed learned machinery therefore remains highly competent at using complete distributed information throughout the tested width ladder. The raw scaling curve is primarily produced by additional required source scope becoming available, while the learned relay computation continues to utilize that information effectively.

## No-communication control

**No-communication exact solve was 0% for all 45 seed × difficulty × population conditions.**

Across those control conditions, bit accuracy remained near chance: 49.56% to 50.57%, with a mean of 50.07%.

Thus simply activating more local records does not solve the relay benchmark under the matched isolated-worker control. Recurrent information transfer is necessary for the tested system.

## Frozen Gate assessment

| Seed | relay-2 endpoint | relay-4 endpoint | relay-8 endpoint | Useful tiers | Communication tiers | Seed Gate |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 99.0 pp | 98.7 pp | 96.7 pp | 3/3 | 3/3 | **PASS** |
| 2 | 99.9 pp | 99.4 pp | 98.5 pp | 3/3 | 3/3 | **PASS** |
| 3 | 100.0 pp | 99.9 pp | 99.6 pp | 3/3 | 3/3 | **PASS** |

Every relay tier for every seed also had all **4 / 4** adjacent population steps non-decreasing under the frozen 1-point tolerance.

Final aggregate result:

> **3 / 3 seeds PASS -> Gate v0 POSITIVE.**

## What is now supported

The confirmation supports the narrow causal statement:

> **With learned parameters fixed, the canonical shared-weight relay system reproducibly converts additional reusable runtime neural computation and additional available distributed source scope into additional end-to-end capability. Bounded recurrent communication is required on the tested task.**

The result is substantially stronger than the original seed-0 development observation because:

- confirmation worlds were untouched during development;
- three new training seeds independently reproduced the scaling behavior;
- the benchmark shortcut was removed before confirmation;
- canonical code reproduced the development checkpoint and metrics bit-for-bit before the split was opened;
- the pass rule, seed set, configuration, and 3/3 aggregation were frozen before results became visible;
- all three seeds exceeded the minimum criteria on all three relay tiers.

## What is explicitly NOT supported

The already-qualified serial schedule equivalence result is a hard interpretive constraint.

The repaired relay computation can be serialized through one live learned state while keeping the same source scope and the same `N × relay_hops` learned worker updates, and produces the same mathematical result within floating-point tolerance.

Therefore this confirmation does **not** show:

- that simultaneous wide population state creates extra function-level capability over equal-work serial execution;
- better capability per learned worker update/FLOP than a serial recurrent implementation;
- superiority over a dense model;
- superiority on real workloads;
- language, coding, AGI, or superintelligence scaling;
- that 1,024+ or 100K runtime states will continue the curve.

It establishes **runtime-compute scaling at fixed learned parameters**, not yet a population-organization efficiency advantage.

## Relationship to prior art

Fixed-parameter recurrent test-time-compute scaling and shared-weight message passing already exist in the literature. The architecture-specific research question is therefore no longer whether reused weights can benefit from more computation in principle.

The next question is whether this weak-state population organization offers a useful **work/span/resource frontier**: lower sequential depth or wall-clock latency, useful locality, bounded communication, dynamic activation, or better capability under a fixed latency/hardware budget than simpler serial/recurrent or dense alternatives.

## Next gate

Do **not** jump directly to 1,024 / 4,096 / 16,384 workers merely because Gate v0 passed.

First measure a resource/organization frontier using the same fixed checkpoint/function where possible:

1. total learned work / worker updates;
2. critical sequential depth/span;
3. wall-clock latency and throughput on real local GPU hardware;
4. peak activation/state memory;
5. communication scalars/bytes and memory movement;
6. parallel normalized execution versus the exactly equivalent serial schedule;
7. compiler/runtime mode as a separate systems variable.

Only if width provides a meaningful practical frontier should larger population counts or a benchmark requiring persistent distributed state be promoted.

## Preservation

The original GitHub Actions artifact includes the three model checkpoints, full per-condition JSON, aggregate frozen Gate output, runner provenance, stdout/stderr, and hardware metadata.

This repository result should remain immutable. Any later correction or replication creates a new result version rather than rewriting this confirmation evidence.