# Gate 2 development seed-0 result v0

Status: **DEVELOPMENT OUTCOME D — CLEAN DIRECTIONAL PATTERN; NO GATE VERDICT; CONFIRMATION CLOSED**

Measured experiment head:

`06f359b2bc26bf3130552c0272d89f493abce636`

Training seed: `0`

Target hardware reported by the local runner:

- NVIDIA GeForce RTX 4060 Ti;
- PyTorch `2.9.1+cu130`;
- CUDA runtime `13.0`.

This record is based on the locally generated development artifact analyzed by the pre-result read-only analyzer from PR #88. The raw JSON/ZIP bytes were not independently uploaded into this GitHub record, so the externally reported hashes below are preserved as provenance claims rather than independently rehashed here.

## Preserved external artifact identities

Reported ZIP:

`F:\gate2_persistent_state_capacity_v0_development_seed_0.zip`

Reported SHA-256:

`35fcd046c9dcd1685eb9573e19e21a6867093f7b40fe6366f46235bd2c40aca3`

Reported development JSON SHA-256:

`886af9e28b2a17cfd789fa5cbd78fa9f6571ecdf4eb8006cb146bfd4312610d6`

Reported checkpoint SHA-256:

`a43b3c9a004aa82ebc299c10a8f90e43bd137996bf791a82cfc5ad7156edfc03`

Checkpoint parameter fingerprint:

`7e244a8b051b2ee95fccb64ed2759cabd11f7401de6ee12c517d493d86b846ff`

Learned parameter count:

`21,580`

## Training summary

- steps: `1,000`;
- examples seen: `32,000`;
- initial reported batch loss: `0.763038`;
- final reported batch loss: `0.694403`;
- mean last-50 batch loss: `0.659565`.

The initial/final scalar losses must **not** be interpreted as one homogeneous learning curve. Training cycles deterministically across 12 stable `entity_count × width` conditions. Step 0 and step 999 therefore correspond to different conditions; step 999 lands on condition index 3, `C=64, W=1`, a collision-heavy width-1 control. The held-out matrix below is the relevant development signal.

## Stable-persistent width curves

| Entities | Width | Collision load | Exact solve | Bit accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 1 | 16 | 0.0703 | 0.4922 |
| 16 | 4 | 4 | 0.1055 | 0.5820 |
| 16 | 16 | 1 | 0.1172 | 0.6787 |
| 64 | 1 | 64 | 0.0664 | 0.4980 |
| 64 | 4 | 16 | 0.0664 | 0.5195 |
| 64 | 16 | 4 | 0.1250 | 0.6035 |
| 64 | 64 | 1 | 0.1484 | 0.7236 |
| 256 | 1 | 256 | 0.0469 | 0.4766 |
| 256 | 4 | 64 | 0.0469 | 0.4844 |
| 256 | 16 | 16 | 0.0508 | 0.5059 |
| 256 | 64 | 4 | 0.1094 | 0.5898 |
| 256 | 256 | 1 | 0.1250 | 0.6934 |

Random exact solve for an uninformed 4-bit answer is `1/16 = 0.0625`. The largest stable widths are therefore above that nominal random-answer rate, while the most collision-heavy width-1 conditions remain around chance.

## Primary paired development comparisons

| Comparison | C | Width | Exact-solve delta | 95% paired bootstrap CI | Treatment-only / reference-only |
| --- | ---: | ---: | ---: | ---: | ---: |
| stable width vs width 1 | 64 | 64 | +0.0820 | [+0.0352, +0.1328] | 32 / 11 |
| stable width vs width 1 | 256 | 256 | +0.0781 | [+0.0273, +0.1250] | 30 / 10 |
| stable vs reshuffled locality | 256 | 256 | +0.0742 | [+0.0391, +0.1094] | 21 / 2 |
| stable vs reset state | 256 | 256 | +0.0625 | [+0.0117, +0.1133] | 29 / 13 |

All four preregistered directional development comparisons are positive and all four descriptive paired bootstrap intervals exclude zero on this development seed.

## Largest-width controls

| C | Width | Stable | Reshuffled | Reset |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 16 | 0.1172 | 0.0625 | 0.0586 |
| 64 | 64 | 0.1484 | 0.0703 | 0.0664 |
| 256 | 256 | 0.1250 | 0.0508 | 0.0625 |

The control pattern is directionally consistent across all three entity tiers: stable persistent execution performs best at the largest width, while reshuffled locality and reset-state controls fall back near the random exact-answer regime.

## Development interpretation

Under the pre-result interpretation map, this is **Outcome D — clean directional Gate-2 development pattern**:

- C64/W64 stable > C64/W1;
- C256/W256 stable > C256/W1;
- C256/W256 stable > reshuffled locality;
- C256/W256 stable > reset state;
- paired world-level effects, not aggregate-only differences;
- no change to learned parameter count or information/work budget is introduced by width.

This is the first development evidence in the project where population width improves held-out capability while **inspected information and learned recurrent update count are held fixed**, and where disrupting stable locality/persistence degrades the result.

That is stronger architecturally than Gate 0's relay result, but it remains one development training seed and therefore is not a Gate-2 confirmation result.

## Why confirmation does not open yet

Absolute exact-solve performance remains modest and only one training seed has been observed.

The next clean action is **seed robustness without recipe tuning**:

1. repeat the exact same 1,000-step development recipe on training seeds 1 and 2;
2. use the exact same held-out development worlds/matrix;
3. preserve independent checkpoints/artifacts;
4. run the same read-only analyzer;
5. compare whether the four primary directional effects replicate across training seeds.

Do not change architecture, optimizer, training schedule, widths, worlds, evaluation count, bootstrap procedure or controls before this robustness check.

If the same qualitative pattern survives all three development seeds, the project has a strong reason to freeze the recipe and write the untouched confirmation protocol before exposing confirmation worlds.

If it does not replicate, remain in development and diagnose seed sensitivity rather than promoting seed 0.

## Scientific boundary

This record assigns **no Gate-2 verdict**.

It does not establish:

- confirmation robustness across independent training seeds;
- target-GPU Gate-2 resource advantage versus serial persistent execution;
- general intelligence scaling;
- real-workload advantage;
- scaling above population 256;
- optimal state width/training recipe.

Confirmation remains closed.
