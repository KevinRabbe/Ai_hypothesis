# Gate-3 v0 — Development seed 0 result

## Status

**VALID DEVELOPMENT RESULT — OUTCOME A: NO OR INCOMPLETE BREADTH EFFECT**

This is development-only evidence. It is **not** a Gate-3 confirmation verdict.

The result was produced locally on the frozen Gate-3 v0 development protocol after the first two PowerShell wrapper attempts failed before scientific Python execution. Those wrapper failures did not expose training or evaluation data. The admitted scientific run used the subsequently qualified Windows-PowerShell-safe wrapper.

Measured Git head:

`4fd3519bc816d555a38807320bc4a5d6a99a4ca0`

Observed runtime:

```text
torch = 2.9.1+cu130
CUDA runtime = 13.0
GPU = NVIDIA GeForce RTX 4060 Ti
training seed = 0
learned parameters = 19,873
confirmation_opened = false
```

## Frozen development recipe

```text
optimizer:              AdamW
steps:                  1,200
training batch:         128
learning rate:          3e-4
weight decay:           1e-4
grad clip:              1.0
evaluation worlds/tier: 256
evaluation batch:       64
bootstrap samples:      2,000
score quantization:     1e-3
```

The run completed training in about 4.5 minutes and the 36-cell development evaluation in about 0.6 minutes on the RTX 4060 Ti.

## Complete exact-solve matrix

| Depth | Width | Stable diverse | Collapsed diversity | Reshuffled continuity |
|---:|---:|---:|---:|---:|
| H4 | 1 | 0.2578 | 0.2578 | 0.2578 |
| H4 | 4 | 0.1172 | 0.2578 | 0.1367 |
| H4 | 16 | 0.0430 | 0.2578 | 0.1094 |
| H6 | 1 | 0.0898 | 0.0898 | 0.0898 |
| H6 | 4 | 0.0469 | 0.0898 | 0.0469 |
| H6 | 16 | 0.0273 | 0.0898 | 0.0195 |
| H6 | 64 | 0.0156 | 0.0898 | 0.0117 |
| H8 | 1 | 0.0508 | 0.0508 | 0.0508 |
| H8 | 4 | 0.0469 | 0.0508 | 0.0469 |
| H8 | 16 | 0.0352 | 0.0508 | 0.0352 |
| H8 | 64 | 0.0078 | 0.0273 | 0.0039 |
| H8 | 256 | 0.0000 | 0.0273 | 0.0000 |

The width-1 structural identities hold exactly at all three depths.

## Five preregistered primary comparisons

| Primary comparison | Delta | 95% paired bootstrap CI | Direction |
|---|---:|---:|---|
| H6 stable W64 vs W1 | -0.07421875 | [-0.109375, -0.0390625] | negative |
| H8 stable W256 vs W1 | -0.05078125 | [-0.078125, -0.0234375] | negative |
| H8 stable W256 vs W64 | -0.0078125 | [-0.01953125, 0.0] | non-positive |
| H8 stable W256 vs collapsed W256 | -0.02734375 | [-0.05078125, -0.0078125] | negative |
| H8 stable W256 vs reshuffled W256 | 0.0 | [0.0, 0.0] | null |

The two central breadth comparisons are not merely inconclusive: their complete 95% paired bootstrap intervals are below zero.

At H8:

```text
W1 stable:   13 / 256 solved = 0.0508
W64 stable:   2 / 256 solved = 0.0078
W256 stable:  0 / 256 solved = 0.0000
```

At H6:

```text
W1 stable:   23 / 256 solved = 0.0898
W64 stable:   4 / 256 solved = 0.0156
```

## Independent development audit

The frozen independent analyzer reconstructed the paired statistics from raw per-world solve vectors and returned:

```text
artifact_valid = true
errors = []
scientific_status = DEVELOPMENT_ONLY_NO_GATE_VERDICT
directional_outcome = A_NO_OR_INCOMPLETE_BREADTH_EFFECT
```

Primary deltas reconstructed independently:

```text
h6_w64_vs_w1            = -0.07421875
h8_w256_vs_w1           = -0.05078125
h8_w256_vs_w64          = -0.0078125
h8_stable_vs_collapsed  = -0.02734375
h8_stable_vs_reshuffled =  0.0
```

## Interpretation under the preregistered map

The frozen Gate-3 protocol defined Outcome A as:

> Largest-width stable performance does not exceed W1 or W64 on H8.

and interpreted it as:

> Under this workload and learned scorer, distributing fixed learned work across more simultaneous hypotheses does not improve capability.

Seed 0 satisfies Outcome A strongly.

The result therefore **rejects the intended positive Gate-3 v0 development pattern for this checkpoint and protocol**.

It does **not** establish that population computation in general is false. Gate 0, Gate 1 and Gate 2 remain unaffected. This result specifically shows that naively reallocating a fixed recurrent-update budget from deep refinement toward broader simultaneously retained hypothesis beams is not automatically beneficial, and on this frozen scorer/task it is actively harmful over much of the measured width ladder.

## Mechanistic signal

The width curve is monotone or near-monotone downward across the stable treatment at every depth:

```text
H4: 0.2578 -> 0.1172 -> 0.0430
H6: 0.0898 -> 0.0469 -> 0.0273 -> 0.0156
H8: 0.0508 -> 0.0469 -> 0.0352 -> 0.0078 -> 0.0000
```

The collapsed-diversity control is also informative. At large width it outperforms the stable diverse beam, despite deliberately destroying hypothesis diversity. This is opposite the intended breadth mechanism and suggests that under the frozen work allocation the lost recurrent depth/refinement per live hypothesis dominates any benefit from retaining more possibilities.

The reshuffled control does not rescue the effect; at H8 W256 both stable and reshuffled solve zero worlds.

## Local artifact provenance reported by the frozen runner

Checkpoint SHA-256:

`0e0756ee763a4c30a52446025c3dfc6ed0648f87acfe8eaf86bf8969ed6bcc9a`

Result JSON SHA-256:

`19d9bd9f4c8ad5cdd5f85c7498f3d85e093eacdd8a8755b6f4a93a79a798c1e7`

Independent audit JSON SHA-256:

`7a9e2020e90a1e9d1694c1ffd582728e766fa777944f70eb2db46b98b661035c`

Recursive manifest SHA-256:

`99248d21c45d8feedb6a99592e8d347e126d0e7360ec326a767d88bc0df635ab`

Parameter fingerprint:

`aca4aed31805ae4d0005245c6c692b1dd2bb3061c286efb44f98cd8b6de4e24a`

These hashes are transcribed from the preserved local runner output. Repository recording does not independently possess or rehash the external local artifact bytes.

## Scientific boundary

- Confirmation remains closed.
- This result is not a Gate-3 confirmation verdict.
- No training seed may be replaced or rerun to search for a favorable outcome under the same v0 recipe.
- Any new hypothesis/scorer/work-allocation design must be a new explicitly versioned protocol.
