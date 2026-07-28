# Gate 2 — Persistent-state capacity protocol v0

Status: **FROZEN BEFORE ANY GATE-2 DEVELOPMENT RESULT**

## Research question

> With learned parameters, inspected information, and total learned update count held fixed, can a larger population of persistent runtime neural states improve delayed associative capability by reducing state interference, and can that capability be executed on the target GPU with a useful practical resource frontier?

This protocol is activated because Gate 1 produced a positive practical execution frontier for simultaneous population execution.

Gate 2 is intentionally not another source-coverage relay benchmark. Every population width in one world receives **the same complete observation stream** and performs **the same number of learned updates**. Population width changes only how many persistent runtime state slots the same shared learned machinery can reuse.

## Hypothesis under test

A fixed learned cell can act as reusable local machinery while a larger runtime population supplies additional temporary state capacity.

If that mechanism is real, increasing persistent slot count should reduce interference between independently evolving local traces even though:

- learned parameters do not increase;
- every condition sees exactly the same entities and observations;
- every condition performs exactly the same number of learned cell updates;
- training data and output task are unchanged.

The intended causal variable is **persistent runtime-state organization**, not additional source scope.

## Synthetic workload — delayed keyed traces

Each procedural world contains `C` logical entities.

Each entity has:

- a unique random key;
- a random 4-bit payload;
- four evidence observations, one payload bit per evidence round;
- four later interference observations that contain no new payload information;
- one final delayed query asking for that entity's complete 4-bit payload.

The answer space therefore has 16 exact payload classes.

The query occurs only after all evidence and interference rounds. The payload is never repeated at query time.

### Entity counts

Frozen difficulty ladder:

- `C = 16`;
- `C = 64`;
- `C = 256`.

### Runtime population widths

For each entity count evaluate every width not exceeding `C`:

- `1`;
- `4`;
- `16`;
- `64`;
- `256`.

This yields:

- C16: widths `1 / 4 / 16`;
- C64: widths `1 / 4 / 16 / 64`;
- C256: widths `1 / 4 / 16 / 64 / 256`.

No larger population is part of Gate 2. Counts above 256 belong to Gate 3 only after Gate 2 earns them.

## Stable parameter-free routing

Each world creates a random permutation of its entity identities.

For a condition with width `W`, entity rank `r` is assigned to runtime slot:

`slot = r mod W`

The mapping is deterministic, balanced, parameter-free, and stable for the complete world.

Consequences:

- every entity is processed at every width;
- target/source coverage is exactly 100% at every width;
- when `W < C`, multiple entities collide into one persistent state slot;
- when `W = C`, each entity receives one persistent state slot;
- population scaling changes state interference/capacity, not information availability.

The final query is routed by the same deterministic mapping. The queried key is still supplied to the learned readout so a collided slot must recover the requested association rather than simply emit its latest value.

## Learned system

Use one shared learned neural cell and shared readout for all runtime slots.

Requirements:

- one checkpoint per independently trained seed;
- identical learned weights at every width and control within that seed;
- learned parameter count exactly constant across runtime width;
- no worker-specific learned parameters;
- no learned router;
- no oracle payload or future-query information;
- runtime states are temporary activations, not parameters.

The first development implementation should reuse the smallest coherent shared-cell architecture already available in the population-compute stack where practical. Architecture/parameter count must be frozen before the untouched confirmation run is opened.

## Equal-information / equal-work invariant

For one fixed world with entity count `C`, every width must consume the exact same ordered entity observations.

There are eight learned-update rounds per entity:

- 4 evidence updates;
- 4 interference/retention updates.

Therefore every width performs exactly:

`8 × C`

learned cell updates per world before final readout.

Population width must not change:

- number of entities;
- observations seen;
- payload bits seen;
- interference observations;
- update count;
- query identity;
- checkpoint;
- learned parameter count.

A result violating any of these identities is mechanically invalid and must not be interpreted.

## Primary execution mode — stable persistent population

Each active slot owns one neural state vector that persists across all eight rounds.

Observations routed to that slot update the existing state. State is not reset between rounds.

Multiple entities mapped to the same slot therefore compete for finite neural state capacity.

This is the primary Gate-2 mechanism.

## Causal controls

### 1. Serial persistent schedule

Execute the exact same stable persistent state bank while time-multiplexing learned slot updates serially.

It must preserve:

- the same slot bank;
- the same routing;
- the same observation order semantics;
- the same learned update count;
- the same final decoded output within the frozen numerical-equivalence rule.

Capability difference between parallel and serial persistent execution is a correctness failure. Their purpose is resource comparison, not capability comparison.

### 2. Reshuffled-locality control

Keep the same number of persistent state slots, observations, checkpoint, update count, and per-round slot load, but apply an independently frozen balanced entity→slot permutation on every round.

This breaks stable entity-local state continuity without changing source coverage, state count, or learned work.

At width 1 this control is exactly identical to stable routing and must produce exactly the same result. Width-1 disagreement invalidates the control implementation.

### 3. Reset-state control

Keep width, routing, observations, checkpoint, and learned update count fixed, but reset all runtime slot states at each round boundary.

This preserves instantaneous population size while removing cross-round persistent memory.

The control tests whether any width effect depends on persistent runtime state rather than only wider batched computation.

## Development and confirmation separation

Gate 2 follows the same evidence discipline as Gate 0:

1. mechanics and synthetic fixtures;
2. development training/results;
3. freeze architecture, optimizer, world construction, evaluation matrix, numerical rule, and confirmation decision rule;
4. only then open untouched confirmation worlds and new training seeds.

Development evidence may select a coherent training recipe but may not be reported as confirmation.

Target confirmation should use at least three independently trained seeds unless a mechanics failure terminates the experiment earlier.

## Primary capability metrics

For every entity-count × width × control cell record:

- exact 4-bit payload solve rate;
- per-bit accuracy;
- paired stable-vs-control exact-solve outcomes on identical worlds;
- paired width-vs-width exact-solve outcomes on identical worlds;
- collision load per active slot;
- target entity's slot load;
- learned update count;
- inspected entity/observation count;
- learned parameter count and checkpoint fingerprint.

World-level paired comparisons are primary. Aggregate means alone are not sufficient.

Use exact paired discordance statistics or paired bootstrap confidence intervals where appropriate. Do not assign a scientific verdict from a p-value alone.

## Primary causal comparisons

Interpret in this order:

1. **same-information width effect** — stable persistent population across widths within fixed `C`;
2. **stable-locality effect** — stable vs reshuffled routing at the same `C` and `W`;
3. **persistence effect** — stable persistent vs reset-state control at the same `C` and `W`;
4. **resource effect** — parallel persistent vs output-equivalent serial persistent execution on target hardware.

The strongest Gate-2 result would show all four in the expected direction.

## Target-hardware resource measurements

Only after capability mechanics are qualified, measure the stable persistent function on the actual target GPU.

Primary target remains the local RTX 4060 Ti 16 GB unless hardware changes are explicitly recorded before the run.

For parallel and serial persistent schedules record:

- median/p95/min CUDA-event latency after warmup;
- synchronized throughput;
- host enqueue/orchestration time;
- peak allocated/reserved CUDA memory;
- peak simultaneous neural states;
- total state-bank residency;
- total learned updates;
- host↔device transfers if any;
- synchronization count/placement;
- execution mode/compiler mode.

Compiler/graph execution remains a separate variable. Establish eager behavior first.

## Gate-2 positive evidence rule

Gate 2 is positive only if all mechanics identities pass and untouched confirmation supports a capability/resource frontier attributable to persistent population organization.

At minimum, confirmation must show:

1. all widths process identical information and identical total learned update counts within each entity count;
2. one frozen checkpoint per seed is reused unchanged across widths and controls;
3. for both `C = 64` and `C = 256`, the largest stable-persistent width has a positive paired exact-solve advantage over width 1 with a confidence interval excluding zero;
4. at `C = 256`, stable routing at the largest width has a positive paired exact-solve advantage over the reshuffled-locality control with a confidence interval excluding zero;
5. at `C = 256`, stable persistent state has a positive paired exact-solve advantage over the reset-state control with a confidence interval excluding zero;
6. the output-equivalent parallel persistent schedule provides a useful target-hardware resource frontier relative to its serial persistent schedule without an unacceptable memory failure.

No minimum speedup is preregistered. Resource interpretation uses the complete measured frontier.

A single development seed is never enough for the final Gate-2 claim.

## Negative / redirect outcomes

Treat Gate 2 as negative or unresolved for this substrate if any of the following occur:

- capability does not improve with persistent slot count once source coverage and learned updates are fixed;
- width effects disappear under untouched confirmation seeds;
- stable routing does not outperform same-memory reshuffled routing, leaving locality as an unsupported explanation;
- reset-state execution performs equivalently, showing persistence is unnecessary;
- the serial persistent schedule dominates the practical resource frontier;
- required state memory grows too quickly for the capability gained;
- a simpler fixed-state recurrent baseline explains the same frontier without the population organization.

A negative Gate 2 does not invalidate Gate 0 or Gate 1. It would mean the current shared-cell population architecture has not yet demonstrated organization-specific capability beyond source-scope scaling.

## Explicit non-claims

Even a positive Gate 2 would not establish:

- general intelligence;
- natural-language reasoning advantage;
- real code/research workload advantage;
- benefit beyond 256 runtime states;
- distributed/multi-machine scaling;
- optimal communication architecture;
- optimal compiler/runtime organization;
- superiority over every parameter-matched dense architecture.

Those belong to later gates.

## Gate transition

A confirmed positive Gate 2 earns Gate 3: locating the larger useful population frontier at `1,024 / 4,096 / 16,384` and identifying the first real saturation resource.

Do not activate Gate 3 from Gate-1 speed alone.
