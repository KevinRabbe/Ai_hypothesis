# Repository PR consolidation map v0

Status: **QUALIFIED PLANNING MAP — DO NOT CLOSE ACTIVE/HISTORICAL PRS UNTIL THE CANONICAL CONSOLIDATION LINE IS MERGED AND THE CURRENT GATE-2 DEVELOPMENT ARTIFACT IS SAFE**

Audit base: `06f359b2bc26bf3130552c0272d89f493abce636`

This map classifies currently open pull requests by their role after the fixed-parameter population-compute program became canonical.

The goal is not to erase history. Closing a superseded draft PR only changes project-state visibility; its commits, branch, discussion, and scientific provenance remain available.

The hygiene audit has now mechanically proven that the focused population-compute package/regression suite passes with `ai_hypothesis/runtime`, `ai_hypothesis/large_scope`, and `ai_hypothesis/step02` physically absent from a temporary CI checkout. That proves active-core independence, but actual repository-tree removal remains deferred to the final consolidation branch.

## Keep open during current Gate-2 development

### #87 — Gate-2 persistent-state development

Current measured experiment line.

Keep open until the first development artifact is captured, analyzed, and the next Gate-2 development decision is recorded.

### #88 — Gate-2 result analysis

Read-only result-analysis tooling and pre-result interpretation map.

Keep open while the first development artifact is pending and while its analysis is being consumed.

### #89 — Repository hygiene audit

Independent cleanup/audit line.

Keep open until its qualified cleanup decisions have been transferred into the eventual canonical consolidation PR.

## Preserved future direction; close later after canonical merge

### #85 — Evolutionary organism optimization direction

The unique future-research content is now preserved independently of #85 at:

`docs/future/evolutionary_organism_direction.md`

with a current Gate-1-complete / Gate-2-active status boundary. `docs/future/README.md` indexes the deferred direction.

Therefore #85 no longer needs to remain open merely to prevent loss of the idea. It should still remain open until the consolidation line carrying that preserved document lands on canonical `main`.

After that, close #85 as deferred rather than leaving a docs-only future idea looking like an active implementation gate.

## Preserve branch/history, then close as deferred infrastructure

### #86 — Large-scope result contract

Useful result-validation infrastructure for the old independently weighted large-scope program, but not part of current Gate 2.

Close after the deferred `large_scope` stack has a durable archival home and the canonical consolidation line is merged.

### Open PRs #2–#55 that belong to the persistent-runtime / knowledge-integration / large-scope stack

These PRs built substantial reusable infrastructure: Research Ledger, Work Threads, scheduler/control, evidence/knowledge state, integration hierarchy, indexed materialization, scope coverage, persistent large-scope execution, and qualification.

They are not failed work. They are deferred infrastructure whose current tree footprint is mechanically proven unnecessary for the focused population-compute core.

After the canonical consolidation branch preserves any unique architecture documents and lands:

- close the still-open members of this stack as superseded/deferred;
- link the final canonical archive/consolidation point in each closure comment;
- do not delete their branches solely for cleanup;
- do not rewrite their historical claims.

Representative terminal points are #53 for indexed runtime qualification and #54/#55/#86 for the large-scope benchmark/result-contract line.

## Preserve history, then close as superseded research direction

### #1 — Step-2 minority-rescue diagnostic

This belongs to the earlier independently weighted worker/minority-rescue framing.

The result/question remains historical background, but it is no longer the primary architecture.

Close after the canonical repository explicitly records that earlier independently trained worker work is background rather than the current hypothesis.

### Other still-open Step-2 / independently weighted worker adapters in the #1–#14 era

Preserve their history and any unique evidence. Close after consolidation because the current organism uses shared/reused learned machinery instead of growing learned capacity with worker count.

## Preserve scientific lineage, then close after canonical consolidation

The following open PRs are important Gate-0/Gate-1 history and should **not** be treated as disposable implementation noise:

- #56 — fixed-parameter population-compute reframing;
- #58 — scope/capability decomposition;
- #59 / #60 — early relay training runners;
- #64 — compositional relay repair;
- #74 — relay-v1 answer-frontier benchmark repair;
- #76 — clean relay-v1 development evidence;
- #77 — exact serial-control construction/equivalence;
- #78 — canonical repaired relay-v1 protocol;
- #81 — frozen multi-seed Gate-0 confirmation evidence;
- #82 — positive Gate-1 v1 target-GPU resource frontier.

These PRs may be closed only after the eventual canonical line contains their permanent protocols/result records and the repository entry point links those records directly.

Closure reason should be **consolidated into canonical research history**, not simply "obsolete".

The result documents, version boundaries, failed preregistrations, and provenance remain permanent evidence even after the PR UI is closed.

## Already-closed diagnostic/test-only PRs

Closed diagnostic/test-only PRs such as the relay diagnostic/confirmation execution branches should remain closed. Do not reopen them merely to make the stack look linear.

Their relevant evidence should be referenced from permanent result records rather than from active PR status.

## Closure order after Gate-2 development artifact is safe

Completed already on the audit line:

- population-compute import-boundary proof;
- physical-removal simulation of deferred Python stacks;
- full focused population-compute qualification after that removal;
- CI ownership narrowing;
- preservation of the unique #85 evolutionary direction;
- open-PR lifecycle classification.

Remaining sequence:

1. Capture and preserve the first Gate-2 development artifact/provenance.
2. Create the canonical consolidation branch from the appropriate current scientific line.
3. Preserve canonical Gate-0/Gate-1/Gate-2 result/protocol documentation.
4. Remove/relocate deferred runtime/large-scope/Step-2 active-tree packages under the already-proven dependency boundary.
5. Update README/roadmap/current status.
6. Re-run complete consolidation qualification on the final tree.
7. Merge the consolidation line so `main` becomes truthful again.
8. Close deferred runtime/large-scope/Step-2 PRs with canonical/archive links.
9. Close historical Gate-0/Gate-1 stack PRs with canonical result links.
10. Close #85 after its preserved future-direction document exists on canonical `main`.
11. Leave only genuinely active Gate-2 work and new independent experiments open.

## Standard closure language

For historical scientific lineage:

> Consolidated into the canonical population-compute research line. This PR remains part of the scientific/provenance history; its protocol/result evidence is preserved and linked from the canonical repository. Closing the draft changes project-state visibility only and does not invalidate or delete the recorded work.

For deferred infrastructure:

> Deferred from the current population-compute execution core after dependency/qualification checks showed it is not required by the active gate. The branch and history remain available for later reactivation; closing this draft avoids presenting deferred infrastructure as active research.

For superseded research direction:

> Superseded as the primary research direction by the fixed-learned-parameter shared-weight population-compute program. The earlier result remains useful background and is preserved in repository history.

For preserved future direction:

> Preserved on the canonical deferred-research line and intentionally not active at the current gate. Closing this draft keeps project state truthful while retaining the research direction for later activation if earlier evidence justifies it.

## Hard rule

Do not mass-close PRs before the canonical consolidation branch is merged and the current Gate-2 development artifact/provenance is safe.
