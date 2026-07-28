# Repository hygiene audit v0

Status: **QUALIFIED CONSOLIDATION AUDIT; NO CANONICAL PACKAGE DELETION OR PR CLOSURE YET AUTHORIZED**

Audit base:

`06f359b2bc26bf3130552c0272d89f493abce636`

This audit exists to make the current neural-population research line easier to understand and maintain without erasing the substantial earlier research history that led to it.

## Executive finding

The repository now has one clear active scientific program but several historical implementation programs still living in the same active tree and pull-request stack.

The main practical problems are:

1. `main` is no longer the effective research baseline. The Gate-2 line is hundreds of commits ahead while `main` has only a no-net-content accidental-temp-file add/remove divergence.
2. Dozens of historical stacked draft PRs still appear active even when their question has been superseded or deferred.
3. The current Gate-2 line carries older persistent-runtime, large-scope, and Step-2 implementation stacks that are not part of the current Gate-2 experiment.
4. Historical CI ownership was too broad, causing unrelated systems to be requalified on current research PRs.
5. The README and roadmap still describe Gate 1 as active even though Gate 1 is closed positive and Gate 2 is the active development gate.

None of these findings imply that the older work should be destroyed. Git history, result evidence, protocol history, and potentially reusable infrastructure are different things from the current canonical implementation surface.

## Qualification completed by this audit

Two increasingly strong dependency checks now pass on GitHub CI.

### 1. Static import boundary

`tests/test_population_compute_dependency_boundary.py` AST-scans every module in `ai_hypothesis/population_compute` and rejects imports from:

- `ai_hypothesis.runtime`;
- `ai_hypothesis.large_scope`;
- `ai_hypothesis.step02`.

This check passes.

### 2. Temporary physical-removal simulation

The repository-hygiene workflow then performs a stronger proof in its disposable CI checkout:

```text
rm -rf ai_hypothesis/runtime
       ai_hypothesis/large_scope
       ai_hypothesis/step02
```

After those packages are physically absent, CI:

1. compiles the complete `ai_hypothesis/population_compute` package;
2. runs the canonical population-compute contract tests;
3. runs collective-relay, shared-cell, compositional/reset-state and relay-model tests;
4. runs Gate-0 relay training/confirmation regressions;
5. runs Gate-1 serial/resource/equivalence/precision/audit regressions;
6. runs Gate-2 persistent-state/model/development regressions.

That full focused suite passes.

This is stronger than an import grep: it demonstrates that the current population-compute Python research core and its focused executable qualification do not require the deferred runtime, large-scope, or Step-2 packages to exist in the checkout.

### Interpretation boundary

This result supports **removal eligibility from the future canonical active tree** for those deferred Python packages.

It does not mean:

- their historical code or Git history should be destroyed;
- every old document or benchmark should be discarded;
- later gates cannot reactivate useful ideas from them;
- the running Gate-2 experiment branch should be modified now.

Actual canonical-tree deletion/relocation remains a separate consolidation action after the current Gate-2 development artifact and provenance are safe.

## Proposed repository classes

### A. Canonical active research

Keep directly on the current canonical line:

- `ai_hypothesis/population_compute/` current shared-weight population-compute implementation;
- Gate-0/1/2 scientific contracts required for reproducibility;
- `experiments/population_compute_scaling_v0/` canonical protocols/result records;
- Gate-1 and Gate-2 local runners/finalizers;
- focused population-compute tests;
- top-level hypothesis, roadmap, research questions and scientific-discipline documentation;
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

Useful as an archive/future-source pool, but now mechanically proven unnecessary for the active population-compute Python core:

- `ai_hypothesis/runtime/` Research Ledger / Work Thread / scheduler / knowledge-integration stack;
- `ai_hypothesis/large_scope/` large-scope relevance and persistent-runtime benchmark stack;
- associated runtime/integration/index/consolidation documents;
- associated large-scope/persistent-scope benchmark documents;
- associated focused tests and CI lanes.

This work is not judged bad or useless. It is simply a different abstraction layer from the current small scientific substrate. Parts may become relevant again for dynamic activation, richer search/reasoning workloads, persistent evidence integration, or evolutionary organism work.

The completed physical-removal simulation means these packages may be removed from the future canonical active tree without breaking the currently qualified population-compute suite, subject to the final consolidation qualification.

### D. Superseded research direction

Preserve in history but do not present as the current architecture:

- old Step-2 independently weighted worker/minority-rescue machinery;
- earlier independently trained tiny-worker framing;
- abandoned/provisional reducer and evidence-rescue experiments that predate the fixed-parameter shared-weight hypothesis.

Their results remain useful background for learned local-transform size and failure analysis, but they should not dominate the current repository entry point.

### E. Deferred future ideas

Keep as durable research-direction documents rather than active implementation stacks until an earlier gate earns them:

- evolutionary organism/lineage optimization;
- learned scheduler/routing;
- compiler-specific execution studies;
- very large 1K+ population execution;
- multi-machine/geographic distribution.

The evolutionary-organism direction from historical PR #85 is being preserved on the hygiene/consolidation line so that the future idea does not depend on an otherwise deferred PR remaining open.

## CI ownership — implemented on the hygiene line

The following ownership cleanup is now staged and qualified:

### Population-compute CI

Owns current shared-weight population research, including Gate 0/1/2 and the useful structural relay regressions.

The canonical lane now includes the old compositional-relay and population-state-reset structural tests, so retiring the historical compositional training diagnostic from generic PRs does not remove those correctness invariants.

### Indexed-runtime CI

Uses path ownership for `ai_hypothesis/runtime/**` and its focused indexed-runtime tests/workflow.

### Large-scope CI

Uses path ownership for `ai_hypothesis/large_scope/**` and focused large-scope tests/workflow.

### Historical compositional-relay CI

No longer has a generic pull-request trigger. It remains available on its historical branch and by manual dispatch.

This does not weaken current scientific qualification. It removes repeated execution of unrelated historical systems while moving still-relevant relay correctness checks into the canonical population-compute lane.

## Pull-request lifecycle map

`docs/repository_pr_consolidation_map_v0.md` classifies the currently open PR stack before any closures occur.

The key policy is:

- keep #87, #88 and #89 open during current Gate-2 development;
- preserve the unique evolutionary direction from #85 before closing it later;
- treat the old runtime/large-scope chain as deferred infrastructure, not failed work;
- treat old independently weighted Step-2 work as superseded background;
- preserve Gate-0/Gate-1 PRs as scientific lineage until their permanent evidence is directly represented from canonical `main`;
- close superseded drafts only after consolidation, with successor/evidence links.

Closing a superseded PR is project-state cleanup, not deletion of its scientific history.

## `main` consolidation candidate

The long-term clean shape should make `main` the truthful repository entry point again.

Do not force-reset `main` during the running Gate-2 experiment.

After the Gate-2 development result is captured and repository cleanup is ready:

1. create one consolidation branch from the current scientific line;
2. retain canonical population-compute code/evidence;
3. remove/relocate the deferred packages now proven unnecessary for the active scientific core;
4. preserve selected deferred-research documents;
5. update README/roadmap/current status;
6. qualify the complete resulting tree again;
7. merge that single consolidation PR into `main`;
8. then close superseded stacked draft PRs using the recorded lifecycle map.

This preserves old commits in repository history while making the present tree reflect the present research program.

## Documentation drift to repair after the running artifact is safe

The current README correctly states the fixed-parameter computational-organism hypothesis and Gate-0 evidence, but still frames Gate 1 as the immediate/current question.

The current roadmap likewise labels Gate 1 as active and describes Gate 2 only generically.

Canonical status should become:

- Gate 0: **completed positive**;
- Gate 1: **completed positive on RTX 4060 Ti eager CUDA**;
- Gate 2: **active; delayed-keyed-traces development/confirmation program**;
- Gate 3: **locked pending confirmed Gate-2 capability/resource evidence**.

The documentation update is deliberately not placed on the exact measured Gate-2 experiment branch while its first development artifact is still being captured/preserved.

## Explicit non-actions during the current GPU run

Until the first Gate-2 development artifact has completed and its provenance is preserved:

- do not change the Gate-2 experiment branch;
- do not change the running training recipe;
- do not change its protocol or evaluation matrix;
- do not delete result files/stashes;
- do not merge/rebase the local running branch;
- do not open confirmation;
- do not remove deferred packages from that exact branch;
- do not close historical PRs whose canonical successor/evidence link is not yet established.

The hygiene audit runs on a separate branch and is not part of the measured development experiment.
