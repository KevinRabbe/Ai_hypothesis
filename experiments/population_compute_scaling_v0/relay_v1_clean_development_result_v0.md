# Relay v1 clean one-checkpoint development result

## Status

**Development evidence only. Confirmation remains unopened.**

This result comes from the test-only diagnostic PR #75 on top of the qualified `collective-relay-v1-answer-frontier` benchmark from #74.

The diagnostic intentionally reused the #73 experiment script byte-for-byte. The only scientific variable changed from the earlier contaminated run was the relay benchmark generator: the final answer-producing chain edge is now forced to the declared scope frontier so its value cannot appear in a smaller information-incomplete prefix.

## Provenance

- workflow run: `30231455648`
- artifact: `8640303015`
- benchmark: `collective-relay-v1-answer-frontier`
- evaluation split: `development`
- confirmation opened: `false`
- training seed: `0`
- training steps: `2,000`
- training batch size: `64`
- examples seen: `128,000`
- learned parameters: `26,669`
- checkpoint fingerprint: `0ae8f231df258fbad5775edf42796fd79bfe3aea9f675803e242d0221fc8fbdb`
- state width: `64`
- message width: `24`
- gate-supervision weight: `1.0`
- communicating reducer: parameter-free softmax-normalized gate competition
- oracle information used at inference: `false`

Training loss moved from `2.11323` initially to `0.06605` finally; the mean total loss over the final 50 steps was `0.07761`.

The checkpoint was persisted and reloaded before evaluation, and the same learned parameter fingerprint was used across every population condition.

## Raw communicating capability

Exact solve rate using normalized shared communication:

| Active workers | relay-2 | relay-4 | relay-8 |
| ---: | ---: | ---: | ---: |
| 1 | 0.0% | 0.0% | 0.0% |
| 4 | 24.9% | 25.0% | 0.0% |
| 16 | 49.9% | 49.9% | 33.1% |
| 64 | 74.9% | 74.8% | 66.4% |
| 256 | 99.7% | 99.4% | 99.0% |

The raw curve now tracks the benchmark's designed information-availability ladder rather than premature answer exposure.

## Scope availability

Information-complete rate:

| Active workers | relay-2 | relay-4 | relay-8 |
| ---: | ---: | ---: | ---: |
| 1 | 0.0% | 0.0% | 0.0% |
| 4 | 25.0% | 25.0% | 0.0% |
| 16 | 50.0% | 50.0% | 33.3% |
| 64 | 75.0% | 75.0% | 66.7% |
| 256 | 100.0% | 100.0% | 100.0% |

For every evaluated difficulty/population condition where information was incomplete, **exact solve rate given incomplete information was 0.0%**.

This is the key benchmark-v1 integrity result: the answer-frontier repair removed the shortcut that contaminated relay-v0.

## Capability given complete information

Exact solve rate conditional on the full required chain being inside active scope:

| Active workers | relay-2 | relay-4 | relay-8 |
| ---: | ---: | ---: | ---: |
| 4 | 99.6% | 100.0% | — |
| 16 | 99.8% | 99.8% | 99.40% |
| 64 | 99.87% | 99.73% | 99.55% |
| 256 | 99.7% | 99.4% | 99.0% |

The same learned checkpoint therefore remains population-stable across the tested ladder once the information needed by the task is available.

## No-communication control

Exact solve was **0.0% at every population size and every relay depth** under the matched no-communication condition.

Bit accuracy stayed near chance (~49.6-50.7%), including at 256 workers where the full chain is inside scope.

This separates simple scope exposure from recurrent population communication: seeing all local records is not sufficient for the no-communication population to execute the relay chain.

## Development interpretation

This result is clean development evidence for the following statement:

> With one fixed learned parameter set, increasing active runtime population increases end-to-end capability on a task whose required information is distributed across worker-local contexts, while bounded recurrent communication is required to exploit that distributed information.

The causal chain observed in relay-v1 is:

```text
larger active population
    -> larger source scope becomes available
    -> normalized shared communication executes the distributed relay
    -> raw exact capability rises
```

The result also establishes that the learned population computation itself remains effective as width grows: solve-given-complete stays approximately 99-100% across 4/16/64/256 rather than collapsing with population size.

## What this does not establish

This development run does **not** establish:

- frozen-confirmation Gate-v0 success;
- reproducibility across independent training seeds;
- superiority over an equal-compute serial control;
- superiority over a dense model;
- efficiency advantage after GPU/runtime costs;
- general language, coding, reasoning, AGI or superintelligence scaling.

The synthetic relay benchmark tests one architecture property only: whether a fixed learned update system can use more runtime states and bounded communication to solve more distributed work.

## Next causal boundary

Before a broad architecture claim, execute the preregistered serial-compute control with the same learned machinery and matched worker-update budget.

The serial control should expose the same relevant source scope while time-multiplexing it through a small number of recurrent states. Its purpose is to distinguish:

```text
benefit from population state / parallel organization
```

from:

```text
benefit from simply spending more recurrent worker updates
```

After that control is mechanically valid, freeze the repaired canonical training/execution configuration and run untouched confirmation worlds across at least three independent training seeds under the already-frozen Gate criteria.
