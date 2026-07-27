# Serial schedule equivalence result v0

## Status

**Executable architecture result. No confirmation data was opened.**

This result qualifies the preregistered serial-compute control for the repaired normalized relay computation.

It should be interpreted together with the clean relay-v1 development result in `relay_v1_clean_development_result_v0.md`.

## Question

The relay-v1 development curve showed that one fixed 26,669-parameter checkpoint reaches increasing raw capability as active runtime population grows from 1 to 256 states, while solve-given-complete stays approximately 99-100%.

The serial control asks a narrower causal question:

> Does the current relay function require simultaneous wide population-state residency for capability, or can the same learned computation be time-multiplexed through one live neural state when total learned worker updates and source scope are held equal?

## Why exact serialization is possible here

The repaired relay has two properties:

1. each worker's hidden state is reset from its immutable local record at every relay hop;
2. population transport uses a parameter-free softmax-normalized weighted sum of candidate value messages.

Within one hop, every record therefore applies the same learned update independently to:

```text
(local record, current shared query)
```

before the deterministic reducer combines the resulting gate logit and candidate value.

The N record-local updates can be evaluated simultaneously or sequentially without changing the mathematical function.

## Matched schedules

For N active source records and H relay hops:

### Parallel normalized schedule

- N learned record states live simultaneously;
- N learned record updates per hop;
- one normalized population reduction per hop;
- total learned worker updates per sample: `N × H`;
- peak live learned states: `N`.

### Serial normalized schedule

- one learned record state lives at a time;
- the same N source records are processed sequentially for each hop;
- the same learned input projection, GRU update, message gate and value projection are used;
- softmax reduction is accumulated online with a numerically stable running maximum / denominator / weighted numerator;
- total learned worker updates per sample: the same `N × H`;
- peak live learned states: `1`.

The serial schedule changes execution order and state residency only. It does not reduce source scope or learned update count.

## Executable qualification

GitHub Actions population-compute run `30237608647` at head `73b83dc01ec4352d89f9266b52006ac00890aae6` completed successfully.

The qualification artifact `8642143727` reports:

```text
Ran 39 tests in 5.204s
OK
```

The serial/parallel equivalence regression uses arbitrary fixed model weights and covers every combination of:

```text
relay depth:      2 / 4 / 8
active workers:   1 / 4 / 16 / 64 / 256
```

For every condition it requires:

- parallel and serial final shared representations equal within floating-point tolerance;
- parallel and serial final logits equal within floating-point tolerance;
- decoded node predictions exactly identical;
- identical `N × H` learned worker-update counts;
- identical candidate-evaluation counts;
- parallel peak live neural states = N;
- serial peak live neural states = 1;
- unchanged model parameter fingerprint.

The full population-compute contract/regression suite remained green.

## Result

The repaired normalized relay function is **schedule-equivalent at matched learned-update count**.

Therefore the current relay benchmark does not demonstrate additional function-level capability caused specifically by simultaneous wide population-state residency.

The same computation can be executed as:

```text
wide parallel population
```

or

```text
one time-multiplexed learned state + deterministic online reduction
```

without changing the result.

## What the clean population curve still proves

This does **not** invalidate the clean relay-v1 development result.

The relay-v1 evidence still shows that, with learned parameters fixed:

```text
more runtime learned updates + more available source scope
    -> more end-to-end capability
```

and that recurrent information transfer is necessary on the tested distributed task.

The no-communication control remains unable to solve even when all source records are visible.

What changes is the interpretation of **width itself**.

For this repaired relay computation, width currently provides:

- parallel evaluation of independent record-local neural updates;
- lower sequential depth / potentially lower wall-clock latency;
- larger simultaneous state residency;

but it does not provide a different result from an equal-work serial execution.

## Scientific consequence

The current benchmark supports:

> **fixed learned parameters can support capability that scales with additional reusable runtime neural computation and source scope.**

It does not yet support the stronger statement:

> **wide population organization produces more capability per learned worker update than an equivalent serial computation.**

That stronger question requires a benchmark/architecture where the persistent distributed state itself matters, or an explicit resource frontier where parallel width produces a practical latency/throughput advantage under matched hardware constraints.

## Next research boundary

Do not hide this result by inventing more routing machinery.

Before frozen confirmation, the successful repaired training/inference mechanisms should be promoted from diagnostic code into one canonical versioned implementation.

Then the next architecture-specific experiment should distinguish two fronts explicitly:

1. **compute scaling:** capability versus total reusable learned worker updates at fixed learned parameters;
2. **organization scaling:** capability/latency versus width at matched total work or matched wall-clock budget.

A future population-state benchmark should require persistent local information across rounds or interactions that cannot be reduced to independent record-local updates followed by a commutative set reduction without losing the target resource advantage.

Confirmation of relay-v1 can still test reproducibility of the fixed-parameter compute-scaling result, but it must not be described as evidence that simultaneous population state is intrinsically more capable than equal-work serial execution.
