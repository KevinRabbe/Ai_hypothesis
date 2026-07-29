# Gate-2 Post-Confirmation Preparation Plan v0

Status: **PREPARATION ONLY — RUNNING CONFIRMATION HEAD REMAINS UNTOUCHED**

Exact running confirmation head:

`c2a26a17a94746ca88f29950197131689405917b`

This plan defines useful work that may be prepared while the frozen Gate-2 confirmation suite is executing, without changing the measured branch or exposing/using confirmation results prematurely.

## Safe preparation targets

1. Confirmation result auditor
   - consume the suite summary plus all three seed artifacts;
   - verify seeds are exactly 3/4/5;
   - verify 512 confirmation worlds per entity tier;
   - verify confirmation split and opened-confirmation marker;
   - verify one checkpoint identity per seed and no mutation during evaluation;
   - verify 36 cells and 33 paired summaries per seed;
   - verify equal-information/equal-work invariants;
   - verify width-1 stable/reshuffled identity;
   - independently recompute the 12 frozen primary pass/fail decisions from the stored paired summaries;
   - never invent a replacement rule after seeing results.

2. Resource-frontier execution runner
   - use the already-frozen `gate2_resource_frontier_protocol_v0.md`;
   - load the preselected confirmation seed-3 checkpoint;
   - require idle-machine attestation and no Factorio process;
   - correctness preflight before timing;
   - time parallel-persistent vs serial-persistent schedules only under the frozen cells/batches/repeats;
   - save raw timings, provenance, hashes and a manifest;
   - progress reporting is allowed outside measured timing sections.

3. Resource result auditor
   - independently recheck checkpoint/protocol identity;
   - verify decoded schedule identity/correctness preflight;
   - recompute timing summaries from raw samples;
   - evaluate the frozen resource pass rule exactly;
   - distinguish capability confirmation from resource confirmation.

4. Final Gate-2 verdict assembler
   - capability confirmation PASS/FAIL is read only from the frozen capability rule;
   - resource result PASS/FAIL is read only from the frozen resource rule;
   - overall Gate-2 may be positive only if both required components pass;
   - no post-hoc threshold relaxation;
   - negative/mixed outcomes remain permanent evidence.

5. Consolidation readiness
   - prepare, but do not apply, README/roadmap promotion;
   - prepare final Gate-2 evidence links;
   - prepare canonical `main` consolidation after result evidence is safe;
   - do not close historical PRs until canonical evidence exists on the consolidated line.

## Explicit non-actions while confirmation runs

- do not commit to or move the exact running confirmation branch;
- do not change seeds, steps, worlds, batch sizes, model or optimizer;
- do not inspect confirmation seed/world outputs through any alternative path;
- do not run an extra confirmation seed;
- do not replace a failed seed;
- do not change the 12/12 acceptance rule;
- do not time the resource frontier while unrelated GPU-heavy applications are running;
- do not optimize/compile the measured resource implementation because compiler mode is a separate variable.
