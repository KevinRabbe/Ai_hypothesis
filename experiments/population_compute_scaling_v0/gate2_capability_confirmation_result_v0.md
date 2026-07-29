# Gate-2 capability confirmation result v0

Status: **CONFIRMED POSITIVE CAPABILITY RESULT; RESOURCE HALF STILL PENDING**

Frozen confirmation measurement head:

`c2a26a17a94746ca88f29950197131689405917b`

Confirmation protocol:

`gate2-persistent-state-confirmation-v0`

## Confirmation corpus

The frozen confirmation used three new independently trained seeds:

- seed 3;
- seed 4;
- seed 5.

Each seed used:

- 1,000 optimizer steps;
- training batch size 32;
- the unchanged shared-weight Gate-2 model architecture;
- 512 untouched confirmation worlds per entity-count tier;
- the full 36-cell C × W × control matrix;
- 2,000 deterministic paired-bootstrap samples;
- the same fixed information and `8 × C` learned recurrent-update budget per world across compared widths/controls.

Confirmation remained isolated from development seeds/worlds.

## Primary rule

Each training seed independently had to pass all four preregistered paired comparisons with the 95% bootstrap confidence-interval lower bound strictly above zero:

1. C64 stable largest width W64 > stable W1;
2. C256 stable W256 > stable W1;
3. C256 W256 stable locality > reshuffled locality;
4. C256 W256 persistent state > reset state.

No pooling, seed replacement, or 2-of-3 rule was allowed.

This is therefore a 12-comparison confirmation requirement.

## Independent raw-world audit

After all three neural runs completed, the original PowerShell wrapper hit a packaging-only StrictMode bug while counting zero failed seeds. All seed checkpoints/results/manifests had already been written. The bug was not in training, evaluation, world construction, or neural computation.

A no-rerun finalizer reconstructed only the top-level confirmation suite from the completed immutable seed artifacts.

An independent auditor then ignored the runner's final summary booleans and reconstructed the scientific result from the per-world `solved_by_world` vectors. It independently checked:

- exact measurement head and clean-tree provenance;
- exact seeds 3/4/5 and frozen recipe;
- 36 conditions per seed;
- 512 confirmation worlds and world ordering;
- checkpoint/result/runtime SHA and parameter-fingerprint consistency;
- fixed information/work invariants;
- width-1 stable/reshuffled exact identity;
- all 33 paired comparisons per seed;
- deterministic 2,000-sample bootstrap intervals from the raw paired outcomes;
- stored paired summaries against the raw-world recomputation;
- all four primary decisions per seed;
- the final 12/12 capability rule.

Independent audit result:

```json
{
  "artifact_valid": true,
  "capability_confirmation_passed": true,
  "errors": [],
  "seed_passes": {
    "3": true,
    "4": true,
    "5": true
  }
}
```

## Primary 95% CI lower bounds

Every frozen primary lower bound was strictly positive.

| Seed | C64 W64 > W1 | C256 W256 > W1 | C256 stable > reshuffled | C256 stable > reset |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 0.04296875 | 0.068359375 | 0.056640625 | 0.052734375 |
| 4 | 0.03125 | 0.0546875 | 0.05078125 | 0.041015625 |
| 5 | 0.029296875 | 0.044921875 | 0.0546875 | 0.048828125 |

Thus all **12 / 12** preregistered primary paired comparisons passed independently.

## Checkpoint identities

Seed 3:

- checkpoint SHA-256: `8c5c03df82b3c43e67a51f9169d5a7e8ca8348215aeb46d72bb580a27d6bf7c2`
- parameter fingerprint: `d19636886d394fbe8e4b49503f4202df803c1af1705be11e7638a71568cbd354`

Seed 4:

- checkpoint SHA-256: `6a4c8f07cacbd27ba23309542c3d95f977d3b89f69626a4416959dab2a8caaa1`
- parameter fingerprint: `21c0e0b2640e3972258a5b83a11051128d0ef4f46f50d0d2b403b877f0afd570`

Seed 5:

- checkpoint SHA-256: `f50c696278ce70c14108f15d8bc7f91bc64fc58ae4503f02ccb3d5abd62d9879`
- parameter fingerprint: `7d45fea4de9be535230a1e50a6a9bd1aa029a89735430cf44044aa960874d59a`

Each model contained 21,580 learned parameters.

## Local preservation hashes

Before top-level finalization, the completed confirmation directory was preserved as:

`F:\gate2_confirmation_v0_before_finalization.zip`

Reported SHA-256:

`02dc99218f39ef9fd003821da39e1d3ea4b9176056c911eda04971db34d739d6`

The independent audit JSON was preserved as:

`F:\gate2_confirmation_v0_independent_audit.json`

Reported SHA-256:

`51ac0374eca1621df3f047a2eb52a8f1abc30b0b45ac9650606e3c4447318c5b`

These local archive/audit hashes are recorded provenance supplied from the target machine. The repository does not contain those binary/local artifacts.

## Scientific conclusion

The frozen Gate-2 capability hypothesis is **confirmed positive on the delayed-keyed-traces workload**:

> With learned parameters, inspected information, and total learned recurrent-update count held fixed, increasing the number of stable persistent runtime neural states reproducibly improves delayed associative capability. The confirmed advantage depends on both stable locality and persistence: disrupting locality by reshuffling or erasing state by reset significantly reduces the largest-width result on C256.

This is stronger than Gate 0 because the compared widths receive the same inspected entities/observations and the same learned recurrent-update count. The observed width effect is therefore not attributable to extra source coverage or extra learned work.

## Boundaries

This result does **not** establish:

- overall Gate-2 v0 positivity yet;
- a useful practical resource frontier versus the matched serial-persistent schedule;
- per-FLOP or energy superiority;
- general reasoning/search scaling;
- optimal state width/model architecture;
- useful scaling beyond 256 runtime states;
- compiler/graph execution benefits;
- multi-machine behavior.

The preregistered Gate-2 v0 definition requires the separate frozen RTX 4060 Ti eager-CUDA resource half to pass as well.

Until that resource result is independently audited, the overall Gate-2 v0 verdict remains:

**NOT ASSIGNED — CAPABILITY HALF CONFIRMED POSITIVE, RESOURCE HALF PENDING.**
