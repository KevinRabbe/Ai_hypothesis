# Gate-3 v1 robustness development seed 1

Status: **VALID ROBUSTNESS DEVELOPMENT RESULT — OUTCOME C**

Scientific status: robustness development only; no Gate-3 verdict. Confirmation remained closed.

Measured execution head: `4bcc4a7032d05e082f7b5d5f6d19ebbd041b81c3`

Training seed: `1`
Learned parameters: `19,649`
Parameter fingerprint: `2a57984f755ee1404fd828c2df36a5efff8f13a541b7d7c93a891014fbf4897c`

## Stable exact coverage

| Tier | L1 | L4 | L16 | L64 | L256 |
|---|---:|---:|---:|---:|---:|
| S6 | 0.1367 | 0.4023 | 0.4023 | — | — |
| S8 | 0.0859 | 0.2656 | 0.5977 | 0.5977 | — |
| S10 | 0.0273 | 0.1172 | 0.4766 | 0.6094 | 0.6094 |

## Frozen primary paired effects

| Comparison | Delta | 95% paired bootstrap CI |
|---|---:|---:|
| S8 stable L64 vs L1 | +0.51171875 | [0.453125, 0.578125] |
| S10 stable L256 vs L1 | +0.58203125 | [0.515625, 0.64453125] |
| S10 stable L256 vs L64 | 0.0 | [0.0, 0.0] |
| S10 stable L256 vs collapsed L256 | +0.58203125 | [0.51953125, 0.64453125] |
| S10 stable L256 vs reshuffled L256 | +0.51171875 | [0.44140625, 0.58203125] |

Independent audit:

- `artifact_valid = true`
- `errors = []`
- `directional_outcome = C_LATENT_CAPACITY_HELPS_BUT_SATURATES_EARLY`

## Interpretation

Seed 1 independently reproduces the Gate-3 v1 seed-0 mechanism pattern. Under fixed learned parameters, fixed two-child active neural width, fixed eight recurrent updates per evaluated child, fixed search rounds, fixed total learned recurrent work, and identical development worlds, increasing dormant persistent-hypothesis capacity strongly increases no-replay exact solution coverage.

The population mechanism remains control-separated: the collapsed logical-one control stays at the L1 result, and reshuffling persistent state/score continuity removes most of the stable-reserve advantage.

The useful frontier again saturates by L64 on S10. L64 and L256 are exactly tied on all 256 paired development worlds (`delta = 0`, bootstrap interval `[0,0]`). This plateau is preserved as evidence and is not tuned away.

Seed 1 therefore counts as a successful independent robustness replication of **Outcome C**, not as confirmation and not as a positive Gate-3 verdict.

## Provenance

- checkpoint SHA-256: `8ce384627f5543fa4fe78498f9990f8214cf2a4afed0c9b734e86761ad13c989`
- result SHA-256: `8525272080af7121b3c427a94be77e00b714d2248ab58ddf9e6b228154604eae`
- audit SHA-256: `94f06266ebbf5400e4a89878ec5b6adcc018b40cb32bd4417d692de15daa1271`
- manifest SHA-256: `94b88791f716c6d469d1891ed5847141997e3fc0e096215874b1fbf9234d1967`

The previously frozen robustness policy remains unchanged: exactly seed 2 is the remaining authorized robustness checkpoint; confirmation remains closed.
