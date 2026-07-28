# Repository hygiene audit v0

Status: **READ-ONLY CLASSIFICATION; NO CODE OR EVIDENCE DELETION AUTHORIZED**

Audit base:

`06f359b2bc26bf3130552c0272d89f493abce636`

This audit exists to make the current neural-population research line easier to understand and maintain without erasing the substantial earlier research history that led to it.

## Executive finding

The current repository has one clear scientific program but several historical implementation programs living in the same active tree and pull-request stack.

The main practical problems are:

1. `main` is no longer the effective research baseline. The Gate-2 line is hundreds of commits ahead while `main` has only a no-net-content accidental-temp-file add/remove divergence.
2. Dozens of historical stacked draft PRs still appear active even when their question has been superseded or deferred.
3. The current Gate-2 line carries older persistent-runtime, large-scope, and Step-2 implementation stacks that are not part of the current Gate-2 experiment.
4. Legacy CI workflows run on every pull request because they have unconditional `pull_request` triggers, so Gate-2 changes repeatedly qualify unrelated historical systems.
5. The README and roadmap still describe Gate 1 as active even though Gate 1 is now closed positive and Gate 2 is the active development gate.

None of these findings imply that the older work should be destroyed. Git history, result evidence, protocol history, and potentially reusable infrastructure are different things from the current canonical implementation surface.

## Proposed repository classes

### A. Canonical active research

Keep directly on the current canonical line:

- `ai_hypothesis/population_compute/` current shared-weight population-compute implementation;
- Gate-0/1/2 scientific contracts required for reproducibility;
- `experiments/population_compute_scaling_v0/` canonical protocols/result records;
- Gate-1 and Gate-2 local runners/finalizers;
- focused population-compute tests;
- top-level hypothesis, roadmap, research questions and construction/scientific-discipline documentation;
- current population-compute CI.

This is the implementation line that answers the current question:

> Can fixed learned machinery turn additional reusable runtime neural state/population computation into additional capability under controlled information/work budgets?

### B. Canonical historical evidence

Keep permanently accessible even when no longer part of active execution:

- Gate-0 frozen confirmation protocol/result;
- Gate-0 serial-equivalence evidence;
- Gate-1 v0 failed numerical preregistration and diagnostics;
- Gate-1 v1 frozen precision-aware protocol/result;
- external archive identity and packaging provenance;
- benchmark version boundaries such as relay-v0 -> relay-v1 answer-frontier repair;
- failed/negative development evidence that motivated protocol changes.

Historical evidence should not be deleted simply because later code supersedes it.

### C. Deferred reusable infrastructure

Currently useful as an archive/future-source pool, but not required by Gate 2 unless a later gate explicitly reactivates it:

- `ai_hypothesis/runtime/` Research Ledger / Work Thread / scheduler / knowledge-integration stack;
- `ai_hypothesis/large_scope/` large-scope relevance and persistent-runtime benchmark stack;
- associated `docs/*runtime*`, integration/index/consolidation documents;
- associated `benchmarks/large_scope_*` and persistent-scope benchmark documents;
- associated focused tests and CI lanes.

This work is not judged bad or useless. It is simply a different abstraction layer from the current small scientific substrate. Parts may become relevant again for Gate 4+ dynamic activation, richer search/reasoning workloads, persistent evidence integration, or evolutionary organism work.

Before removal from the canonical tree, the dependency audit must prove that `ai_hypothesis.population_compute` has no import dependency on these packages and a clean-tree experiment branch must pass the population-compute qualification without them.

### D. Superseded research direction

Preserve in history but do not present as the current architecture:

- old Step-2 independently weighted worker/minority-rescue machinery;
- earlier independently trained tiny-worker framing;
- abandoned/provisional reducer and evidence-rescue experiments that predate the fixed-parameter shared-weight hypothesis.

Their results remain useful background for learned local-transform size and failure analysis, but they should not dominate the current repository entry point.

### E. Deferred future ideas

Keep as short research-direction documents rather than active implementation stacks until an earlier gate earns them:

- evolutionary organism/lineage optimization;
- learned scheduler/routing;
- compiler-specific architecture work;
- very large 1K+ population execution;
- multi-machine/geographic distribution.

## Dependency boundary to prove

The first mechanical hygiene contract is:

`ai_hypothesis.population_compute` must not import:

- `ai_hypothesis.runtime`;
- `ai_hypothesis.large_scope`;
- `ai_hypothesis.step02`.

`tests/test_population_compute_dependency_boundary.py` performs an AST-level repository import audit for this exact boundary.

A passing audit does **not** authorize deletion. It only proves that the current scientific core is not implicitly coupled to those older packages.

The stronger future proof is a clean consolidation branch where deferred packages are absent and the full population-compute qualification still passes.

## CI cleanup candidate

Current historical workflows use unconditional pull-request triggers. That makes every current Gate-2 PR execute unrelated indexed-runtime, large-scope, and compositional-relay qualification.

After the current Gate-2 development artifact is safely captured, narrow those workflows with path filters or branch-specific triggers.

Recommended current-CI ownership:

- population-compute changes -> population-compute qualification;
- runtime changes -> indexed-runtime qualification;
- large-scope changes -> large-scope qualification;
- old compositional-relay-specific changes -> relay workflow only when its owned files change;
- repository-level contract changes -> deliberately opt into multiple lanes when needed.

Do not weaken scientific qualification merely to make CI faster; remove only irrelevant repeated qualification.

## Pull-request hygiene candidate

The repository contains many historical stacked draft PRs whose code is already inherited by later branches.

After the current Gate-2 development result is preserved:

1. identify the final successor/canonical branch for each historical PR chain;
2. ensure any unique deferred-direction document is preserved on a durable branch or copied into the canonical deferred-research docs;
3. close superseded draft PRs with a short final comment linking their successor/canonical evidence;
4. do not delete branches or PR discussions solely for cleanliness;
5. keep only current experiments and genuinely independent future directions open.

Closing a superseded PR is project-state cleanup, not deletion of its scientific history.

## `main` consolidation candidate

The long-term clean shape should make `main` the truthful repository entry point again.

Do not force-reset `main` during the running Gate-2 experiment.

After the Gate-2 development result is captured and repository cleanup is qualified:

1. create one consolidation branch from the current scientific line;
2. retain canonical population-compute code/evidence;
3. remove or relocate only proven-unneeded active-tree packages while preserving Git history;
4. update README/roadmap/current status;
5. qualify the complete resulting tree;
6. merge that single consolidation PR into `main`;
7. then close superseded stacked draft PRs.

This preserves all old commits in repository history while making the present tree reflect the present research program.

## Documentation drift to repair

The current README correctly states the fixed-parameter computational-organism hypothesis and Gate-0 evidence, but still frames Gate 1 as the immediate/current question.

The current roadmap likewise labels Gate 1 as active and describes Gate 2 only generically.

Once the running development artifact is captured, update canonical status to:

- Gate 0: **completed positive**;
- Gate 1: **completed positive on RTX 4060 Ti eager CUDA**;
- Gate 2: **active; delayed-keyed-traces development/confirmation protocol**;
- Gate 3: **locked pending confirmed Gate-2 capability/resource evidence**.

## Explicit non-actions during the current GPU run

Until the first Gate-2 development artifact has completed and its provenance is preserved:

- do not change the Gate-2 experiment branch;
- do not change the running training recipe;
- do not change its protocol or evaluation matrix;
- do not delete result files/stashes;
- do not merge/rebase the local running branch;
- do not open confirmation;
- do not close historical PRs whose unique content has not yet been classified.

The hygiene audit runs on a separate branch and is not part of the measured development experiment.
