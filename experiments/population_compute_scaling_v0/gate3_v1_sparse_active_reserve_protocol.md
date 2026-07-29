# Gate-3 v1 — Sparse active search with latent hypothesis reserve

## Status

**FROZEN DEVELOPMENT PROTOCOL — NO GATE-3 v1 RESULT INSPECTED**

Gate-3 v1 is a new protocol motivated by the negative Gate-3 v0 development result. It does not modify, rescue, reinterpret, or reopen Gate-3 v0.

Gate-3 v0 established that distributing a fixed recurrent-update budget across every live hypothesis simultaneously caused performance to degrade as width increased. Gate-3 v1 tests a different mechanism:

> Can a larger population of already-evaluated dormant hypothesis states improve search capability when expensive neural activation remains sparse, per-evaluated-hypothesis refinement is fixed, learned parameters are fixed, and total learned recurrent work is fixed?

The core distinction is:

```text
v0: larger population => less neural refinement per live hypothesis
v1: larger reserve    => same refinement per evaluated child,
                         but more evaluated alternatives may remain dormant
```

## Scientific question

Within each difficulty tier, hold fixed:

- learned parameter count and checkpoint;
- hidden-world distribution;
- noisy evidence process;
- expensive active neural lanes per search expansion;
- recurrent updates per evaluated child;
- total search-expansion rounds;
- total learned recurrent-update count;
- deterministic search policy;
- score quantization / tie breaking;
- training recipe and evaluation worlds.

Vary only the maximum number of distinct already-evaluated nonterminal hypothesis states that may remain in the latent reserve.

The primary causal question is:

> With the same expensive learned work, does retaining more dormant alternatives increase the probability that best-first neural search eventually generates the exact hidden solution?

## Why v1 follows v0

Gate-3 v0 showed that breadth and per-hypothesis refinement cannot be collapsed into one variable: the largest v0 populations received dramatically less recurrent refinement per hypothesis and performed worse.

Gate-3 v1 therefore decouples:

1. **latent population capacity** — how many evaluated alternatives can be remembered without further work;
2. **active learned computation** — how many hypotheses are processed now;
3. **total learned work** — how much expensive neural computation the whole organism receives.

Increasing latent capacity in v1 does not reduce the neural refinement assigned to a child when that child is evaluated.

## Workload — noisy binary solution search

Each world contains a uniformly sampled hidden binary path.

Frozen difficulty tiers:

| Tier | Hidden depth `D` | Solution space | Search rounds `E` | Reserve capacities `L` |
|---|---:|---:|---:|---|
| S6 | 6 | 64 | 16 | 1, 4, 16 |
| S8 | 8 | 256 | 64 | 1, 4, 16, 64 |
| S10 | 10 | 1,024 | 256 | 1, 4, 16, 64, 256 |

A world also contains one noisy binary hint for each path depth.

Frozen hint reliability:

`P(hint[d] == hidden_bit[d]) = 0.70`

Hints are generated deterministically from the world seed and are identical for every reserve-capacity condition on that world.

The runtime never sees the hidden path.

## Search hypothesis representation

A nonterminal hypothesis contains:

- candidate-path prefix metadata;
- one persistent neural state vector;
- one scalar neural priority score;
- its current prefix depth.

All neural states reuse one shared learned function and one shared parameter set.

There are no learned parameters that scale with reserve capacity.

Candidate metadata exists only to define routing identity, tree depth, branch generation and final evaluation instrumentation. It may not inspect or compare against the hidden answer during runtime.

## Active learned expansion

Each productive search round activates exactly one nonterminal parent and exactly two child-evaluation lanes.

Procedure:

1. select the highest-priority nonterminal hypothesis from the latent reserve;
2. remove it from the reserve;
3. create branch-0 and branch-1 children by cloning the parent neural state and appending the branch bit to metadata;
4. provide each child with the noisy hint corresponding to the new child depth plus its proposed branch-action token;
5. apply exactly `K = 8` recurrent neural updates to **each** child;
6. produce one neural priority score for each child;
7. terminal children are recorded by instrumentation and retired from the search frontier;
8. nonterminal children enter the latent reserve;
9. if the reserve exceeds `L`, retain the top `L` distinct hypotheses by the frozen score/tie-break rule.

Thus every evaluated child receives exactly eight recurrent updates regardless of `L`.

No large-reserve condition may create a cheaper or lower-refinement neural child than a small-reserve condition.

## Fixed active lanes

The expensive active neural width is fixed at two child lanes per productive expansion.

`L` is **not** the active neural batch size.

A reserve of 256 dormant hypotheses therefore does not imply 256 simultaneous expensive neural evaluations.

The runtime may store dormant state vectors, candidate metadata and cached scalar scores without additional learned recurrent computation.

## Total learned-work budget

Each search round has a fixed learned-work budget:

`2 children * 8 recurrent updates = 16 learned recurrent updates`

Frozen total learned work:

| Tier | Search rounds | Learned updates / round | Total learned recurrent updates |
|---|---:|---:|---:|
| S6 | 16 | 16 | 256 |
| S8 | 64 | 16 | 1,024 |
| S10 | 256 | 16 | 4,096 |

These totals are identical across every reserve capacity and control mode within the tier.

The S10 maximum therefore uses the same 4,096-update order as the largest Gate-3 v0 workload while allocating work sparsely rather than across all live states.

## Frontier exhaustion and matched-work sink

Small reserves can irreversibly commit to one path and exhaust their nonterminal frontier before the frozen search-round budget is consumed.

When no nonterminal hypothesis remains, the remaining rounds execute a **matched-work sink**:

- exactly two sink lanes;
- exactly eight recurrent updates per sink lane;
- a fixed answer-independent sink token;
- no world observation;
- no generated candidate;
- no state or score returned to the search reserve.

Sink work exists only to preserve exact learned-work equality after a search has exhausted all retained alternatives.

The artifact must report separately:

- productive expansion rounds;
- sink rounds;
- productive learned updates;
- sink learned updates.

Sink work is not counted as useful search progress.

Because the inability to retain an alternative can make later compute unusable for search, productive-work fraction is a mechanistic outcome of reserve capacity and must be reported rather than hidden.

## Search policy

The runtime uses deterministic best-first activation.

At every productive round, select the highest-priority nonterminal candidate currently present in the reserve.

No reserve-capacity-specific scheduler is permitted.

The scorer cannot receive `L` as an input feature.

## Score ordering

Gate-3 v1 inherits the pre-result numerical lesson from Gate-3 v0.

Raw neural scores remain FP32.

For search ordering only:

`q(score) = round(score / 0.001)`

Candidates are ordered by:

1. descending `q(score)`;
2. deterministic SHA-based answer-independent tie break from world seed, expansion index and candidate-path identity.

The score quantum is frozen before development data.

## Primary outcome — exact search coverage

The primary capability outcome is:

> Was the exact hidden full path generated as a terminal child at any point within the frozen search budget?

The hidden answer is used **only after execution by evaluation instrumentation** to score the transcript.

The runtime itself:

- cannot compare a candidate path with the hidden answer;
- cannot stop early because the correct answer was generated;
- cannot restore a discarded prefix because it was correct;
- cannot receive oracle success feedback during the search.

This makes the metric a direct measure of useful problem-space coverage under a fixed expensive neural budget.

Secondary outcomes include:

- first expansion round at which the correct solution was generated;
- whether the correct prefix was present in the reserve after each productive round;
- maximum and mean distinct reserve population;
- productive-vs-sink work fraction;
- number of unique terminal hypotheses generated;
- deepest correct prefix generated;
- per-depth correct-prefix survival.

## Controls

### 1. Stable reserve — primary treatment

Each distinct candidate retains its own persistent neural state, candidate identity and cached score while dormant.

### 2. Collapsed-diversity reserve

Same physical reserve allocation, same expansion budget, same child-evaluation work and same world evidence.

After every productive expansion/prune, all retained reserve slots are replaced by copies of the current highest-priority candidate identity/state.

This destroys distinct alternative hypotheses while retaining the same nominal reserve/state-bank allocation.

At `L=1`, stable and collapsed are exact structural identities.

### 3. Reshuffled-state continuity

Same candidate identities, reserve capacity, learned work and observations.

After every productive expansion/prune, persistent neural states and cached neural scores are deterministically permuted among retained candidate identities using an answer-independent permutation.

Candidate paths remain distinct, but their neural histories are no longer attached consistently to the same hypothesis.

At `L=1`, stable and reshuffled are exact structural identities.

## Information accounting

All capacity/control conditions on one paired world use the exact same:

- hidden path;
- noisy hint vector;
- hint reliability;
- branch semantics;
- shared learned checkpoint.

A child at depth `d` receives only the frozen world hint `hint[d]` plus its own branch action and public phase/depth features.

Repeated inspection of the same depth hint by multiple candidate evaluations does not create additional unique world information.

Artifacts must report:

- unique hint positions available/inspected;
- total candidate evaluations;
- total recurrent updates;
- productive/sink split.

## Model boundary

Gate-3 v1 uses one shared recurrent scorer.

Frozen architectural requirements before implementation:

- persistent state width 64;
- scalar neural priority score;
- branch action, noisy hint, depth and search-stage features only;
- no reserve-capacity embedding;
- no learned slot identity;
- no attention over the reserve;
- no learned parameters scaling with `L`;
- eager PyTorch baseline;
- no `torch.compile`, CUDA graphs, custom fusion or mixed precision in the admitted capability run.

The exact parameter count and input encoding are frozen in the separate development-recipe record before seed-0 development execution.

## Training semantics

Training teaches one shared scorer to prioritize candidate prefixes under the noisy-hint process.

Training may use hidden paths from the training domain to construct supervised prefix-consistency / ranking targets.

Training must not depend on reserve capacity and must not train a separate policy for different `L`.

Controls remain evaluation-only.

The exact training recipe, model parameterization, loss, candidate sampler, optimizer and number of steps must be frozen before first development evidence.

## Evaluation domains

Deterministic non-overlapping world domains:

- training: seeds below `2^30`;
- development: seeds starting at `2^30`;
- confirmation: seeds starting at `2^31`.

Gate-3 v1 confirmation remains mechanically closed during development.

## Development matrix

Evaluate every valid `(D, L)` cell under all three runtime modes.

Matrix size:

- S6: 3 capacities x 3 modes = 9;
- S8: 4 capacities x 3 modes = 12;
- S10: 5 capacities x 3 modes = 15;
- total = 36 cells/checkpoint.

Development world count and bootstrap count are frozen in the development-recipe record before seed 0.

## Preregistered primary development comparisons

The primary directional comparisons are:

1. S8 stable `L64 > L1` exact search coverage;
2. S10 stable `L256 > L1`;
3. S10 stable `L256 > L64`;
4. S10 stable `L256 > collapsed L256`;
5. S10 stable `L256 > reshuffled L256`.

All primary comparisons use identical paired worlds.

The first three test latent reserve capacity without reducing per-child learned refinement.

The last two test whether any benefit specifically depends on maintaining distinct persistent alternatives rather than nominal state-bank capacity or disconnected neural histories.

## Development interpretation map

Development remains non-confirmatory.

### Outcome A — no latent-reserve benefit

Largest-capacity stable search does not improve over `L1` and `L64` on S10.

Interpretation: simply retaining more dormant evaluated alternatives is not sufficient under this scorer/search policy.

### Outcome B — capacity helps but diversity/continuity controls do not separate

Stable reserve capacity improves coverage, but collapsed or reshuffled controls match stable.

Interpretation: the effect is not specifically attributable to a population of distinct persistent neural hypotheses.

### Outcome C — latent capacity helps but saturates early

Stable reserve beats `L1` and controls, but S10 `L256` does not improve over `L64`.

Interpretation: sparse-active reserve population helps, but the useful capacity frontier saturates at or below 64 under this budget.

### Outcome D — clean sparse-active population pattern

All five preregistered primary directions are positive, including S10 `L256 > L64` and both population controls.

Interpretation: under fixed expensive learned work and fixed per-child neural refinement, retaining a larger dormant population improves search coverage and the effect depends on distinct persistent hypothesis states.

Outcome D is development evidence only and cannot assign a Gate-3 confirmation verdict.

## Structural validity requirements

Every admitted artifact must prove:

- learned parameter count identical across capacities/modes;
- checkpoint fingerprint identical across capacities/modes;
- exactly two child-evaluation lanes per productive expansion;
- exactly eight recurrent updates per evaluated child;
- exact total learned recurrent updates equal within tier;
- exact search-round count equal within tier;
- score quantization identical;
- `L1 stable == L1 collapsed == L1 reshuffled` exactly;
- runtime never compares candidate identity against hidden answer;
- no discarded hypothesis is restored;
- controls receive no extra world information;
- confirmation remains closed.

Failure of any structural invariant invalidates the corresponding scientific result.

## Stop / continuation policy

The first admitted development run uses exactly one training seed defined in the later frozen recipe.

After that first result:

- a strongly negative preregistered pattern may close v1 development without seed shopping;
- an ambiguous pattern requires a robustness policy frozen before additional seeds;
- a clean positive pattern still requires a separately frozen multi-seed robustness rule and later untouched confirmation protocol before any positive Gate-3 claim.

No v1 development outcome may rewrite Gate-3 v0.

## Resource/compiler boundary

This is a capability protocol.

Compiler, CUDA-graph, fusion, scheduling and other execution optimization remain separate experimental variables. They cannot rescue or redefine a negative v1 capability result.
