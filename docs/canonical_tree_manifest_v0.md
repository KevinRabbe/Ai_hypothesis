# Canonical Tree Manifest v0

Status: **CONSOLIDATION TARGET — NOT YET APPLIED TO THE RUNNING GATE-2 LINE**

Qualified audit base: `06f359b2bc26bf3130552c0272d89f493abce636`

This manifest defines the intended repository shape after the current Gate-2 development artifact and provenance are safe.

It separates three different preservation requirements:

1. **active executable surface** — code required by the current scientific program;
2. **directly visible scientific record** — protocols/results/failures that should remain easy to inspect from `main`;
3. **Git-historical/deferred material** — useful earlier implementations that do not need to remain in the active checkout.

The hygiene CI has already proven that the focused population-compute suite passes after `ai_hypothesis/runtime`, `ai_hypothesis/large_scope`, and `ai_hypothesis/step02` are physically removed from a temporary checkout.

## 1. Canonical active Python package

Keep:

```text
ai_hypothesis/population_compute/
```

This package owns the current learned substrate and Gate-0/1/2 mechanics:

- fixed-parameter contracts;
- collective relay benchmark/model/training;
- canonical repaired relay-v1 protocol;
- serial-equivalent relay schedules;
- Gate-1 resource-frontier measurement/audit/diagnostics;
- Gate-2 persistent-state-capacity benchmark;
- Gate-2 persistent shared-weight model;
- Gate-2 development training/evaluation.

Do not fold the old persistent Research Ledger runtime back into this package merely to preserve old architecture work. Reintroduce such mechanisms later only when a measured workload requires them.

## 2. Canonical experiment record

Keep directly reachable from `main`:

```text
experiments/population_compute_scaling_v0/
```

Within this experiment family, preserve versioned protocols and result records even when the corresponding implementation is no longer the current path.

Particularly important permanent evidence includes:

- Gate-0 fixed-parameter protocol history;
- relay benchmark repair history;
- canonical relay-v1 confirmation protocol/result;
- serial-schedule equivalence result;
- Gate-1 v0 failed CUDA correctness preregistration;
- Gate-1 precision/equivalence interpretation boundary;
- Gate-1 v1 frozen precision-aware protocol;
- Gate-1 v1 target-GPU result/provenance;
- Gate-2 frozen development/protocol/execution-semantics documents;
- future Gate-2 confirmation protocol/result when created.

A failed preregistration or negative result is scientific evidence and is not cleanup material.

### Historical diagnostic scripts

A diagnostic script under the experiment directory may remain when it is needed to reproduce/understand a recorded scientific transition.

It should not automatically remain part of generic CI merely because it exists.

## 3. Canonical local execution scripts

Keep current scripts required to reproduce admitted/local results, including:

```text
scripts/run_gate1_resource_frontier.ps1
scripts/run_gate1_resource_frontier_v1.ps1
scripts/finalize_gate1_resource_frontier_v1_existing.ps1
scripts/run_gate2_development.ps1
```

Later Gate-2 confirmation scripts may be added only after the confirmation protocol is frozen.

Historical runners that reproduce superseded development programs may be removed from the active tree once their result/protocol record is durable.

## 4. Canonical focused tests

Keep tests that qualify the active package and preserved scientific contracts.

Core groups include:

```text
tests/test_population_compute_contract.py
tests/test_collective_relay.py
tests/test_shared_population_cell.py
tests/test_relay_population_model.py
tests/test_compositional_relay_protocol.py
tests/test_population_state_reset_policy.py
tests/test_relay_experiment.py
tests/test_relay_serial_control.py
tests/test_relay_experiment_v1.py
tests/test_run_relay_scaling_v1.py
tests/test_confirmation_gate_v1.py
tests/test_relay_resource_frontier.py
tests/test_relay_resource_equivalence_diagnostic.py
tests/test_relay_precision_diagnostic.py
tests/test_relay_resource_frontier_v1.py
tests/test_relay_resource_audit.py
tests/test_gate2_persistent_state_capacity.py
tests/test_gate2_persistent_model.py
tests/test_gate2_development.py
```

The exact list may evolve with Gate 2, but the rule is stable: tests remain because they protect a current/preserved contract, not merely because the implementation once existed.

## 5. Canonical CI

Keep one primary population-compute qualification lane.

It should own:

- `ai_hypothesis/population_compute/**`;
- population-compute experiment protocol/result mechanics where executable validation is useful;
- Gate-1/Gate-2 local-runner parsing/provenance guards;
- current focused tests.

The historical compositional relay structural invariants have already been folded into this lane.

### Temporary hygiene lane

The repository-hygiene workflow is useful through consolidation because it proves independence from deferred stacks.

After those stacks are actually absent from canonical `main`, decide whether to:

- retire the temporary removal-simulation workflow; or
- replace it with a lightweight architectural boundary test preventing accidental resurrection/cross-import without an explicit design decision.

Do not keep redundant CI indefinitely simply because it was useful during migration.

## 6. Canonical top-level documentation

Keep and update:

```text
README.md
docs/hypothesis.md
docs/research_questions.md
docs/roadmap.md
```

After the running Gate-2 development artifact is safe, these should state the actual current sequence:

- Gate 0 — completed positive;
- Gate 1 — completed positive on RTX 4060 Ti eager CUDA;
- Gate 2 — active organization-specific persistent-state development;
- Gate 3 — locked pending confirmed Gate-2 evidence;
- dynamic activation / information-transport / evolution remain downstream questions.

The repository entry point should not require readers to reconstruct current status from dozens of stacked PRs.

## 7. Preserved future research

Keep:

```text
docs/future/README.md
docs/future/evolutionary_organism_direction.md
```

Future high-potential ideas belong in concise durable documents until activated by evidence.

A deferred idea should not require an open implementation PR merely to remain remembered.

Additional future documents may later preserve:

- persistent evidence/integration architecture worth reusing;
- large-scope/search benchmark ideas worth revisiting;
- compiler execution ablations;
- learned/deterministic routing questions.

## 8. Deferred Python packages eligible to leave the active tree

The following are mechanically proven unnecessary for the focused current population-compute core:

```text
ai_hypothesis/runtime/
ai_hypothesis/large_scope/
ai_hypothesis/step02/
```

Target action after Gate-2 artifact safety:

- remove them from the canonical active checkout in the consolidation PR;
- preserve their commits/branches/PR discussions in Git history;
- preserve selected high-value conceptual summaries directly in `docs/history/` or `docs/future/`;
- do not describe the removal as invalidating the older work.

## 9. Deferred tests eligible to leave the active tree

Tests whose only purpose is to qualify removed runtime/large-scope/Step-2 implementations should leave with those implementations.

Examples by family:

```text
tests/test_runtime_*.py
tests/test_incremental_*.py
tests/test_indexed_*.py
tests/test_integration_*.py
tests/test_thread_*.py
tests/test_large_scope_*.py
tests/test_step02_*.py
```

Before final deletion, verify that no test in one of these wildcard families protects a still-active shared contract. Move such a test to the active package's focused suite rather than silently dropping the invariant.

That exact process has already been applied to the old compositional/state-reset relay regressions.

## 10. Deferred workflows eligible to leave the active tree

After their owned packages are absent from canonical `main`, the corresponding implementation-specific workflows should normally be removed rather than retained as dead CI:

```text
.github/workflows/indexed-runtime-ci.yml
.github/workflows/large-scope-benchmark-ci.yml
.github/workflows/compositional-relay-protocol-ci.yml
```

The old compositional workflow contains a historical training diagnostic; its scientific interpretation belongs in experiment/result history rather than requiring a permanent active CI lane.

## 11. Deferred benchmark/doc bulk

The old runtime/large-scope program produced many detailed architecture and benchmark documents.

Keeping every implementation note directly on future `main` would continue to obscure the current research program.

Target policy:

- retain a compact historical/deferred summary of reusable ideas;
- retain any document containing unique scientific evidence not reproduced elsewhere;
- remove implementation-specific planning documents from the active checkout when their code is removed;
- rely on Git history/PRs for full archaeological detail.

Documents should be retained because they carry unique evidence or future design value, not because deleting a file feels destructive.

## 12. Old independently weighted worker program

The independently trained tiny-worker / Step-2 direction remains useful background but is not the current architecture.

Canonical documentation should preserve only the evidence needed to understand the transition:

- useful local transformations survived in a small parameter regime;
- roughly 25K–100K parameters was the earlier useful-worker region, with ~50K used as a practical reference;
- independent checkpoints caused learned capacity to grow with population size;
- this motivated the cleaner shared-weight fixed-parameter experiment;
- minority-evidence/reducer work remains historical failure-analysis/background rather than the current gate.

The old implementation itself does not need to remain in the canonical active package tree.

## 13. Pull-request state after consolidation

The PR UI should reflect current research rather than preserve every stacked implementation layer as “open”.

After the canonical consolidation line is merged:

- close deferred runtime/large-scope/Step-2 PRs with preservation/successor links;
- close Gate-0/Gate-1 construction/result PRs only after their evidence is directly linked from canonical docs;
- close docs-only future-direction PRs after their documents are on canonical `main`;
- leave current Gate-2 and genuinely independent new experiments open.

Use `docs/repository_pr_consolidation_map_v0.md` as the closure plan.

## 14. Final-tree qualification requirement

The actual consolidation PR must re-prove the repository after physical deletion/relocation, not rely only on this preview audit.

Required minimum checks:

1. compile all canonical Python modules;
2. run the complete focused population-compute suite;
3. parse/provenance-check local Gate-1/Gate-2 scripts;
4. validate all canonical evidence/document links;
5. verify no canonical module imports removed namespaces;
6. verify no canonical CI references removed test/module paths;
7. verify README/roadmap status agrees with preserved result records;
8. keep Gate-2 confirmation closed unless independently frozen/activated by its scientific protocol.

Only that final qualified tree should become canonical `main`.
