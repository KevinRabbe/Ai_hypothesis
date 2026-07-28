# Gate 2 development seed-robustness result v0

Status: **DEVELOPMENT ROBUSTNESS PASSED; CONFIRMATION STILL CLOSED**

Measured experiment code head for all three runs:

`06f359b2bc26bf3130552c0272d89f493abce636`

Training seeds:

`0 / 1 / 2`

All three runs used the same development recipe frozen before seeds 1 and 2:

- 1,000 optimizer steps;
- training batch size 32;
- 256 held-out development worlds per entity tier;
- evaluation batch size 64;
- 2,000 paired bootstrap samples;
- CUDA execution;
- unchanged architecture, optimizer, training-condition cycle, evaluation matrix and control semantics;
- development split only;
- confirmation closed.

## Robustness rule

The preregistered development robustness check required every seed to preserve the same positive direction for all four primary comparisons:

1. C64/W64 stable > C64/W1;
2. C256/W256 stable > C256/W1;
3. C256/W256 stable > reshuffled locality;
4. C256/W256 stable > reset state.

Seed-level bootstrap intervals were descriptive development diagnostics and were not required to exclude zero for robustness passage.

## Result

The robustness rule passed on **3 / 3 independent training seeds**.

Stronger than required, every one of the 12 seed × primary-comparison paired bootstrap intervals also excluded zero.

### Primary paired effects

| Training seed | C64 W64 vs W1 | C256 W256 vs W1 | C256 stable vs reshuffled | C256 stable vs reset |
| ---: | ---: | ---: | ---: | ---: |
| 0 | +0.0820 `[+0.0352,+0.1328]` | +0.0781 `[+0.0273,+0.1250]` | +0.0742 `[+0.0391,+0.1094]` | +0.0625 `[+0.0117,+0.1133]` |
| 1 | +0.0820 `[+0.0469,+0.1211]` | +0.0703 `[+0.0312,+0.1094]` | +0.0820 `[+0.0469,+0.1172]` | +0.0664 `[+0.0156,+0.1172]` |
| 2 | +0.0938 `[+0.0469,+0.1484]` | +0.1172 `[+0.0703,+0.1641]` | +0.0977 `[+0.0586,+0.1406]` | +0.1133 `[+0.0664,+0.1641]` |

Across the three development seeds, descriptive mean exact-solve deltas were:

- C64 largest stable width vs W1: **+0.0859**;
- C256 largest stable width vs W1: **+0.0885**;
- C256 stable vs reshuffled locality: **+0.0846**;
- C256 stable vs reset state: **+0.0807**.

These cross-seed means are descriptive only and are not treated as one pooled statistical sample.

## Stable largest-width capability

Largest-width stable exact solve was also consistent across seeds:

| Training seed | C64/W64 stable | C256/W256 stable |
| ---: | ---: | ---: |
| 0 | 0.1484 | 0.1250 |
| 1 | 0.1406 | 0.1367 |
| 2 | 0.1523 | 0.1562 |

Descriptive three-seed means:

- C64/W64 stable exact solve: **0.1471**;
- C256/W256 stable exact solve: **0.1393**.

Random exact solve for an uninformed 4-bit answer is `1/16 = 0.0625`.

## Seed 1 artifact identities

Reported external ZIP SHA-256:

`5a9cf94098d2f53d00f5614b2755490d610fc27aafdc072b98d9f013a61bd96e`

Reported development JSON SHA-256:

`c2991612e2bd8ffca57309e9c92bf20cff68541ef4c2cf0f96e5b47fc0d4e0f2`

Reported checkpoint SHA-256:

`5768eab628f90d1cbc47f1a5215df5b79966402571ebe5ede90bf94fb1de5365`

Parameter fingerprint:

`3c856dc92b451b8338f273f9fbf39e6bb783dfbcd1871489531e9db4febe255a`

## Seed 2 artifact identities

Reported external ZIP SHA-256:

`a4723ef0c06e7941ee075c80f08c42eb0832e859a8ea5f0607cff50d289a234f`

Reported development JSON SHA-256:

`ae36a51e2600cf86ce49a954a6ac99003b0015d1718e26e9e279e6e9eb624a98`

Reported checkpoint SHA-256:

`3a40ec670e156c73404a529b29a138b5756c95c99d085c7355a913b9cab63a2b`

Parameter fingerprint:

`10d402648a556cf3c08fdc40478f06a3434cded32d7a8f1de18449732bfc049b`

Seed-0 artifact identities remain recorded in `gate2_development_seed0_result_v0.md`.

The raw seed-1/seed-2 ZIP bytes were not independently rehashed through GitHub; the hashes above preserve the user-reported local provenance.

## Environmental note

During the local development execution window, the user reported that Factorio was running for part of the time. Exact overlap with individual development seeds is not fully established.

These runs therefore must **not** be used for target-GPU latency, throughput, power or utilization claims.

The development capability results remain useful because the scientific outcome is based on exact held-out predictions under deterministic model/world semantics, not on timing. Any future confirmation/resource run must use an intentionally idle machine and preserve GPU-process/environment provenance.

## Interpretation

This is a robust development-level replication of Outcome D:

- persistent population width improves held-out capability at matched inspected information and matched learned recurrent-update count;
- stable locality matters;
- persistent state matters;
- the direction reproduces across three independently trained checkpoints.

This is stronger than the seed-0 observation and justifies freezing a final Gate-2 confirmation protocol **before confirmation exposure**.

It still does not assign a Gate-2 verdict.

## Next action

Per the frozen seed-robustness plan:

1. freeze the final architecture/training/evaluation recipe;
2. freeze untouched confirmation training seeds that are not 0, 1 or 2;
3. freeze confirmation world count and exact acceptance rule;
4. freeze the target-GPU parallel-vs-serial persistent resource protocol separately;
5. only then open confirmation.

No development tuning is justified between this robustness pass and protocol freeze unless the project deliberately abandons this recipe and starts a new development version.
