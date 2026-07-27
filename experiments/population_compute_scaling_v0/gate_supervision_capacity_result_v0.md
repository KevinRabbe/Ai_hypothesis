# Gate-Supervised Relay Capacity Diagnostic v0 — Development Result

## Status

Development-only fixed-width diagnostic over corrected #64. No confirmation data was opened and no inference architecture was changed.

Workflow run: `30229931573`

Fresh relay-2 checkpoints were trained independently at widths 4 / 16 / 64 / 256. All used the same 26,669-parameter architecture and the same training-only gate objective from #69. Batch sizes 64 / 16 / 4 / 1 kept active worker-state evaluations per optimizer batch fixed at 256.

## Result

| Active workers | Sparse exact | Bit accuracy | Hop1 gate top-1 | Hop2 model-query top-1 | Hop2 clean-query top-1 | Shared→clean cosine | No-comm exact |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.99899 | 0.0% |
| 16 | 99.8047% | 99.9674% | 100.0% | 100.0% | 100.0% | 0.99746 | 0.0% |
| 64 | 98.0469% | 99.6257% | 100.0% | 99.4141% | 100.0% | 0.99236 | 0.0% |
| 256 | **0.0%** | 51.0417% | **100.0%** | **4.6875%** | **100.0%** | **0.33115** | 0.0% |

### Gate margins

- width 4 hop1 / model-hop2 / clean-hop2: `+8.606 / +8.458 / +8.767`;
- width 16: `+7.468 / +7.309 / +7.502`;
- width 64: `+6.321 / +5.670 / +6.081`;
- width 256: `+4.657 / -9.089 / +4.689`.

At width 256, the correct hop-1 worker is still ranked first on every held-out world, and the correct hop-2 worker is also ranked first on every world when given the oracle-clean intermediate query. The same hop-2 gate collapses under the model-produced query to mean rank `70.62 / 256`.

### Shared-query fidelity

Mean hop-1 model-produced shared field versus exact clean next-query representation:

- width 4: cosine `0.99899`, RMSE `0.04111`;
- width 16: cosine `0.99746`, RMSE `0.04326`;
- width 64: cosine `0.99236`, RMSE `0.07342`;
- width 256: cosine **`0.33115`**, RMSE **`0.90263`**.

## Frozen interpretation

The preregistered outcomes were:

1. strong performance through 256 -> selectivity training scales;
2. gate quality degrades with width -> population selectivity remains width-limited;
3. gates remain strong but final solve degrades -> shared-field accumulation/readout is the next bottleneck.

The observed result is case 3.

Training-only gate supervision successfully teaches **relative worker selectivity through width 256**. The width-256 failure occurs after that selection signal: independent sigmoid-gated emissions are summed across the active population, and residual nonmatch emission contaminates the shared query enough that the next hop no longer sees the intended node representation.

This explains the earlier pattern:

> more workers did not fail because the correct worker could not be identified; they failed because the communication reducer did not convert that identification into a population-size-stable shared signal.

The no-communication control remains 0% exact at every width.

## Next diagnostic

Train one width-256 gate-supervised checkpoint under the same frozen protocol and, **without retraining**, compare the ordinary independent-sigmoid sum against a parameter-free normalized competitive reducer using the same learned gate logits and candidate messages.

Measure:

- correct-gate mass versus total nonmatch gate mass;
- standard shared-query fidelity;
- normalized shared-query fidelity;
- hop-2 gate ranking from each produced query;
- ordinary versus normalized two-hop exact solve using the same checkpoint/readout.

Interpretation:

- normalized aggregation restores query fidelity and relay solve -> population-size-dependent residual message accumulation is the primary remaining bottleneck;
- normalized aggregation restores query fidelity but not solve -> output/readout is coupled to the old aggregation distribution;
- normalized aggregation does not restore the query -> candidate-message geometry or a deeper communication representation issue remains.

No Gate-v0 conclusion is claimed.
