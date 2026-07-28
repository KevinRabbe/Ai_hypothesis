# Repository PR consolidation map v0

Status: **PLANNING ONLY — DO NOT CLOSE ACTIVE/HISTORICAL PRS UNTIL THE CANONICAL CONSOLIDATION LINE IS QUALIFIED**

Audit base: `06f359b2bc26bf3130552c0272d89f493abce636`

This map classifies currently open pull requests by their role in the repository after the fixed-parameter population-compute program became canonical.

The goal is not to erase history. Closing a superseded draft PR only changes project-state visibility; its commits, branch, discussion, and scientific provenance remain available.

## Keep open during current Gate-2 development

### #87 — Gate-2 persistent-state development

Current measured experiment line.

Keep open until the first development artifact is captured, analyzed, and the next Gate-2 development decision is recorded.

### #88 — Gate-2 result analysis

Read-only result-analysis tooling and pre-result interpretation map.

Keep open while the first development artifact is pending and while its analysis is being consumed.

### #89 — Repository hygiene audit

Independent cleanup/audit line.

Keep open until the canonical consolidation plan has been proven and transferred to the eventual consolidation PR.

## Preserve content, then close as deferred future direction

### #85 — Evolutionary organism optimization direction

Unique and useful future-research document.

Before closure, preserve `docs/evolutionary_organism_direction.md` on the canonical line or in a durable future-research directory.

After preservation, close as deferred rather than leaving a docs-only future idea looking like an active implementation gate.

## Preserve branch/history, then close as deferred infrastructure

### #86 — Large-scope result contract

Useful result-validation infrastructure for the old independently weighted large-scope program, but not part of current Gate 2.

Close after the deferred `large_scope` stack has a durable archival home and the current scientific core has been proven independent from it.

### Open PRs #2–#55 that belong to the persistent-runtime / knowledge-integration / large-scope stack

These PRs built substantial reusable infrastructure: Research Ledger, Work Threads, scheduler/control, evidence/knowledge state, integration hierarchy, indexed materialization, scope coverage, persistent large-scope execution, and qualification.

They are not failed work. They are deferred infrastructure whose current tree footprint is not required by the Gate-2 experiment.

After the consolidation branch proves population-compute independence and preserves any unique architecture documents:

- close the still-open members of this stack as superseded/deferred;
- link the final canonical archive/consolidation point in each closure comment;
- do not delete their branches solely for cleanup;
- do not rewrite their historical claims.

Representative terminal points are #53 for indexed runtime qualification and #54/#55 for the large-scope benchmark/audit line.

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

1. Create and qualify the canonical consolidation branch.
2. Preserve canonical Gate-0/Gate-1/Gate-2 result/protocol documentation.
3. Preserve unique future-direction documents such as #85.
4. Prove the active population-compute package/test suite works without deferred runtime/large-scope/Step-2 packages.
5. Merge the consolidation line so `main` becomes truthful again.
6. Close deferred runtime/large-scope/Step-2 PRs with canonical/archive links.
7. Close historical Gate-0/Gate-1 stack PRs with canonical result links.
8. Close #85 after its future-direction document exists on canonical `main`.
9. Leave only genuinely active Gate-2 work and new independent experiments open.

## Standard closure language

For historical scientific lineage:

> Consolidated into the canonical population-compute research line. This PR remains part of the scientific/provenance history; its protocol/result evidence is preserved and linked from the canonical repository. Closing the draft changes project-state visibility only and does not invalidate or delete the recorded work.

For deferred infrastructure:

> Deferred from the current population-compute execution core after dependency/qualification checks showed it is not required by the active gate. The branch and history remain available for later reactivation; closing this draft avoids presenting deferred infrastructure as active research.

For superseded research direction:

> Superseded as the primary research direction by the fixed-learned-parameter shared-weight population-compute program. The earlier result remains useful background and is preserved in repository history.

## Hard rule

Do not mass-close PRs before the consolidation branch is qualified and the current Gate-2 development artifact/provenance is safe.
