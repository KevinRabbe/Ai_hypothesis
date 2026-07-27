# Supervised + Normalized Width-256 Relay Diagnostic v0 — Development Result

## Status

Development-only diagnostic over corrected #64. No confirmation data was opened. A fresh width-256 checkpoint was trained with the two previously localized repairs active together:

1. training-only gate-selection supervision;
2. parameter-free softmax-normalized population aggregation during training and primary inference.

Workflow run: `30230445895`

Learned architecture remained unchanged at **26,669 parameters**.

## Result

Primary normalized held-out path:

- exact solve rate: **81.4453%**;
- bit accuracy: **98.1283%**;
- hop-1 correct-worker top-1: **100%**;
- hop-1 mean gate margin: **+4.622**;
- model-produced hop-2 correct-worker top-1: **99.6094%**;
- hop-2 mean gate margin: **+4.472**;
- first-hop shared→clean next-query cosine: **0.99752**;
- first-hop RMSE: **0.02421**;
- mean correct-worker softmax weight: **96.23%**.

Training losses at the final step:

- normalized relay BCE: `0.17881`;
- auxiliary gate loss: `0.04498`.

Controls using the same trained checkpoint:

- legacy sigmoid-sum exact: **0%**;
- legacy sigmoid-sum bit accuracy: **51.11%**;
- no-communication exact: **0%**;
- no-communication bit accuracy: **50.23%**.

## Interpretation

The preregistered outcomes were:

1. strong width-256 solve -> credit assignment + normalized transport jointly remove the observed failure;
2. strong gates/query but poor solve -> pooled recurrent state or readout remains independently limiting;
3. gates/query degrade -> the two mechanisms interact poorly.

The result is between cases 1 and 2:

- the catastrophic width-256 collapse is removed: **0% -> 81.45% exact**;
- selection and query transport are essentially solved;
- but width 256 still trails the ~98–100% fixed-width results seen at 4/16/64.

Therefore the two repairs are jointly sufficient for substantial width-256 relay capability, but they do **not** close the full width-dependent performance gap.

Because hop selection and first-hop query fidelity are already near-perfect, the residual error is downstream of the localized population-routing failures. Plausible remaining causes include final shared/readout error and training-sample efficiency: the equal-active-state batch protocol gives width 256 only one world per optimizer step, versus 4/16/64 worlds at smaller widths.

## Next research boundary

The architecture-specific question should now return to the primary hypothesis rather than continue micro-tuning independent width-256 checkpoints:

> Can **one shared learned checkpoint**, trained across the frozen population ladder with the two demonstrated repairs, turn additional runtime population into increasing capability while keeping learned parameters fixed?

The next development diagnostic should therefore apply:

- gate-selection auxiliary supervision;
- softmax-normalized population aggregation;
- the existing mixed-population relay training/evaluation protocol;

and measure the full fixed-checkpoint curve over `1 / 4 / 16 / 64 / 256`, including information-complete conditional solve rates and no-communication controls.

If the one-checkpoint curve still collapses at large width, then inspect final shared/readout and sample-efficiency effects. Do not open confirmation yet.
