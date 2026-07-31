# Gate-8 fixed-parameter organism architecture v0

## Status

**EXACT 19,649-PARAMETER SHARED ORGANISM ARCHITECTURE ADMITTED — GRAPH EXECUTION, TRAINING, CHECKPOINTING AND SCIENTIFIC TEST EXPOSURE CLOSED.**

Exact qualified base result head:

`c7f5260189ef9ac1a1beb73596446316631090c7`

This stage admits only the neural worker core and exact parameter accounting. It does not admit the graph scheduler, training procedure, checkpoint selection, benchmark execution, 1B model, or any scientific-test world.

## Architectural unit

One `Gate8SharedWorkerCore` instance is reused across:

- every edge worker;
- every population from 32 through 1,024;
- every recurrent round;
- every training and evaluation condition admitted by later stages.

Increasing the runtime population creates additional worker state, not additional learned parameters.

Each worker step receives only:

```text
inbox message code
local transform ID
public root symbol
source-is-root flag
target-is-query flag
inbox-present flag
round-zero flag
previous 32-value worker state
```

The neural core receives no node label, source-node identity, target-node identity, worker index, population, depth, round index, path membership, or answer information.

## Shared computation

Frozen dimensions:

```text
message vocabulary       = 256 codes = 8 bits
primitive transforms     = 8
output symbols           = 16
feature width            = 12
worker hidden-state width= 32
worker input width       = 40
public worker roles      = 4
```

The 40-value recurrent input is:

```text
12-value inbox-code embedding
12-value transform embedding plus public-role embedding
12-value public root-symbol embedding
4 public runtime flags
```

A shared 32-state `GRUCell` updates each active worker. Three shared heads produce:

- 256 message-code logits;
- one activity logit;
- 16 answer-symbol logits.

The later runtime stage must enforce at most one outbound 8-bit message per active worker per round. This architecture stage exposes logits only; it does not choose, route, count, or deliver messages.

## Exact learned-parameter ledger

| Component | Parameters |
| --- | ---: |
| 256 × 12 message-code embedding | 3,072 |
| 8 × 12 transform embedding | 96 |
| 16 × 12 root-symbol embedding | 192 |
| 4 × 12 public-role embedding | 48 |
| 4 × 32 learned role initial states | 128 |
| GRU input weights: 3 × 32 × 40 | 3,840 |
| GRU recurrent weights: 3 × 32 × 32 | 3,072 |
| GRU biases: 2 × 3 × 32 | 192 |
| 32 → 256 message head | 8,448 |
| 32 → 1 activity head | 33 |
| 32 → 16 answer head | 528 |
| **Total** | **19,649** |

Every parameter participates in the admitted forward computation. There is no padding tensor or unused parameter reserve.

## Population-scaling boundary

The architecture contains no:

- learned node-identity table;
- learned worker-identity table;
- population-specific module;
- depth-specific module;
- per-round learned table;
- per-world learned state;
- separate network copy per worker;
- hidden full-graph encoder.

Runtime hidden tensors scale with the number of workers, but the model object and all learned tensors remain identical. Contract tests instantiate one model and process synthetic batches of 32 and 1,024 workers while proving that parameter identity and count do not change.

## Topology boundary

Node-label equality, mailbox ownership, synchronous round order, sparse activation, outbound-message selection, delivery, communication accounting, terminal readout, and ablation behavior belong to the later runtime contract.

They are intentionally absent here so architecture qualification cannot expose a scientific graph, answer, path, or scheduler implementation.

## Training boundary

This stage contains no:

- optimizer;
- backward pass;
- loss function;
- differentiable discrete-message surrogate;
- training-world generator;
- checkpoint serialization;
- checkpoint-selection rule;
- validation or scientific-test evaluation;
- 1B tokenizer or model load.

The next slice may admit the deterministic graph runtime and synthetic contract-only execution. Training remains a separate subsequent admission boundary.
