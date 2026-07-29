# Gate-3 v1 — Development seed 0 result

## Status

**VALID DEVELOPMENT RESULT — OUTCOME C: LATENT CAPACITY HELPS BUT SATURATES EARLY**

This is development-only evidence. It is not a Gate-3 confirmation verdict.

Measured scientific head:

`254e7994568c78349b23d71c41be61ca0dee7d98`

Runtime:

```text
torch = 2.9.1+cu130
CUDA runtime = 13.0
GPU = NVIDIA GeForce RTX 4060 Ti
training seed = 0
learned parameters = 19,649
confirmation_opened = false
```

The frozen independent analyzer accepted the artifact with:

```text
artifact_valid = true
errors = []
scientific_status = DEVELOPMENT_ONLY_NO_GATE_VERDICT
directional_outcome = C_LATENT_CAPACITY_HELPS_BUT_SATURATES_EARLY
```

## Complete exact-coverage matrix

| Depth | Capacity | Stable reserve | Collapsed diversity | Reshuffled continuity |
|---:|---:|---:|---:|---:|
| S6 | 1 | 0.1367 | 0.1367 | 0.1367 |
| S6 | 4 | 0.4023 | 0.1367 | 0.1211 |
| S6 | 16 | 0.3945 | 0.1367 | 0.0586 |
| S8 | 1 | 0.0859 | 0.0859 | 0.0859 |
| S8 | 4 | 0.2656 | 0.0859 | 0.0508 |
| S8 | 16 | 0.6016 | 0.0859 | 0.1328 |
| S8 | 64 | 0.6016 | 0.0859 | 0.0820 |
| S10 | 1 | 0.0273 | 0.0273 | 0.0273 |
| S10 | 4 | 0.1172 | 0.0273 | 0.0117 |
| S10 | 16 | 0.5039 | 0.0273 | 0.1016 |
| S10 | 64 | 0.6133 | 0.0273 | 0.0898 |
| S10 | 256 | 0.6133 | 0.0273 | 0.1133 |

The stable treatment therefore shows a large capacity benefit before reaching a clear plateau:

```text
S6:  0.1367 -> 0.4023 -> 0.3945
S8:  0.0859 -> 0.2656 -> 0.6016 -> 0.6016
S10: 0.0273 -> 0.1172 -> 0.5039 -> 0.6133 -> 0.6133
```

## Five preregistered primary comparisons

| Primary comparison | Delta | 95% paired bootstrap CI |
|---|---:|---:|
| S8 stable L64 vs L1 | +0.515625 | [0.453125, 0.58203125] |
| S10 stable L256 vs L1 | +0.5859375 | [0.51953125, 0.6484375] |
| S10 stable L256 vs L64 | 0.0 | [0.0, 0.0] |
| S10 stable L256 vs collapsed L256 | +0.5859375 | [0.5234375, 0.6484375] |
| S10 stable L256 vs reshuffled L256 | +0.5 | [0.42578125, 0.5703125] |

Four mechanism comparisons are strongly positive with complete paired intervals above zero. The one failed intended direction is the incremental S10 L256-vs-L64 frontier comparison, which is an exact paired plateau.

At S10:

```text
L1 stable:    7 / 256 covered = 0.0273
L64 stable: 157 / 256 covered = 0.6133
L256 stable:157 / 256 covered = 0.6133
```

For L256-vs-L1 there are 150 treatment-only worlds and zero reference-only worlds. For L256-vs-L64 all 256 paired outcomes are identical.

## Mechanistic interpretation

Gate-3 v0 showed that increasing simultaneous breadth while dividing a fixed recurrent-update budget across every live hypothesis was harmful because per-hypothesis refinement collapsed.

Gate-3 v1 removed that confound: active neural width remained two child lanes, each evaluated child always received eight recurrent updates, total learned recurrent work stayed fixed, and increasing `L` only increased the number of already-evaluated dormant hypotheses that could remain available to future best-first search.

Under that design, latent reserve capacity produced a large held-out coverage gain. The collapsed-diversity control stayed exactly at the L1 coverage level, while the stable reserve strongly outperformed it. The stable reserve also strongly outperformed the reshuffled-continuity control. This supports the specific mechanism that retaining multiple distinct persistent alternatives with consistent neural histories is useful under the frozen no-replay search workload.

The result does **not** support the stronger preregistered claim that increasing capacity from L64 to L256 gives further capability at S10. The useful capacity frontier saturated by L64 for this scorer, search budget and workload.

## Scientific boundary

This result supports the narrower development hypothesis:

> At fixed learned parameters, active neural width, per-evaluated-child refinement and total learned work, a larger dormant population of persistent hypotheses can substantially improve answer-blind search coverage.

It does not establish:

- a positive Gate-3 confirmation verdict;
- benefit beyond L64 under this workload;
- superiority to arbitrary serial algorithms that can replay/recompute discarded branches;
- general problem-solving or intelligence scaling outside this controlled search task.

No confirmation world was opened.

## Preserved local provenance reported by the admitted runner

Checkpoint SHA-256:

`e63748a1182472d03c540f9123c3442ce44e130519e2176681034568826db590`

Result JSON SHA-256:

`26559f5c48ae2971dbb83507afbe9346c6575653e352202ad8b450b739423342`

Independent audit JSON SHA-256:

`289be3ba58c22a9276220804daf7358d97ff5402533cb89e21e1b3c5f53ccf32`

Recursive manifest SHA-256:

`d31fcd36bdaf3416a1926e390d934da926623f34a99497e04340bf1fbc2b773a`

Parameter fingerprint:

`e4f9990f08e85393a155637cfa50b5831d779770447ec3716fc9e67946992afc`

These hashes are transcribed from the preserved local runner output. Repository recording does not independently possess or rehash the external local artifact bytes.
