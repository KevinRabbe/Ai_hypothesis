# Post-Training Learning L0 qualified stack audit v0

## Purpose

This document records the exact qualified, open, draft, unmerged Post-Training Learning L0 implementation stack and the protected handoff order while the Population Language L0 reference run continues.

It prevents later context loss from changing:

- which commit each layer was qualified on;
- the order in which the stacked branches depend on one another;
- the review-before-merge policy;
- the authorization boundaries;
- which operational components are still genuinely missing.

## Merge policy

`REVIEW_BEFORE_MERGE_NO_AUTOMATIC_MERGE`

No entry in this stack is authorized to merge itself. The audit records the branches as open, draft, and unmerged at qualification.

## Exact qualified stack

| Order | PR | Role | Exact base | Qualified head |
|---:|---:|---|---|---|
| 1 | #203 | Frozen scientific protocol | `bfd2111b65f805e6379ad45ecda6f5fe09d2a282` | `48e8edb9ff39417bfb5cb44521318efa032a340a` |
| 2 | #204 | Bounded neural adapter | `48e8edb9ff39417bfb5cb44521318efa032a340a` | `508a1021f3724a39023d4a4f7c6918d98f379f5c` |
| 3 | #207 | Calibration grid and selection contract | `508a1021f3724a39023d4a4f7c6918d98f379f5c` | `19aa701c475b19fc5b31409528948f21ad9fbdf4` |
| 4 | #208 | Deterministic execution primitives | `19aa701c475b19fc5b31409528948f21ad9fbdf4` | `821449afe7381d4becc9c43dc456632b66b8f034` |
| 5 | #209 | Strict reference-checkpoint loader | `821449afe7381d4becc9c43dc456632b66b8f034` | `0b43d2cfedcaaf92a9905750ba3cac809645bebd` |
| 6 | #210 | True subprocess restart boundary | `0b43d2cfedcaaf92a9905750ba3cac809645bebd` | `f0cf83d1be0426fda976f08a379ab040be53ba89` |
| 7 | #211 | Completed-reference output verifier | `f0cf83d1be0426fda976f08a379ab040be53ba89` | `4bb09762948fd83eb7a7ea2beb5b1f8ecdbd450d` |
| 8 | #212 | Hash-pinned non-authorizing calibration plan | `4bb09762948fd83eb7a7ea2beb5b1f8ecdbd450d` | `780c19c0c8e63dafe6e7c74bbfe3d579129e53fe` |
| 9 | #215 | Calibration work manifest and result verifier | `780c19c0c8e63dafe6e7c74bbfe3d579129e53fe` | `52f0e5e1a97a2a78d42b17e6861da4464665c754` |

Every entry's base SHA equals the previous entry's qualified head SHA. The machine-readable manifest verifies this chain.

## Current protected defaults

The following remain false:

- active reference-output access allowed;
- calibration execution authorized;
- final-world access authorized;
- final execution authorized;
- automatic merge allowed;
- 50M architecture frozen;
- 100M architecture frozen;
- 300M architecture frozen.

The reference training process and output directory remain untouched until the run is complete and an explicit output path is supplied.

## Honest remaining operational work

The preparation stack is not the same as a complete scientific executor. Four components remain explicit:

1. **Real authorization-gated calibration row executor**
   - consumes one exact work item;
   - loads one exact checkpoint;
   - trains one exact adapter candidate;
   - evaluates only the assigned calibration world;
   - performs the true fresh-process restart;
   - writes one exact result row;
   - cannot access final worlds.

2. **Final-evaluation candidate lock and plan**
   - can be built only from a valid independently verified calibration selection;
   - remains non-authorizing;
   - contains no final labels.

3. **Final result verifier**
   - independently validates the frozen final result after separately authorized execution.

4. **PowerShell operator runbook**
   - exact branch, SHA, path, hash, disk, environment, interruption, and restart commands;
   - explicit approval points before calibration and final execution.

These components must not be silently treated as complete merely because the surrounding schemas are qualified.

## Protected handoff order

1. wait for reference training to complete;
2. receive the final console output without discovering outputs;
3. verify one explicitly named reference output through the PR #211 boundary;
4. build the hash-pinned calibration plan through the PR #212 boundary;
5. qualify the real calibration row executor;
6. request separate explicit calibration authorization;
7. run only the frozen 144-row calibration;
8. verify calibration through the PR #215 boundary;
9. lock the selected candidate or record a valid rejection;
10. request separate explicit final-execution authorization.

A valid calibration rejection is a complete scientific outcome and blocks final evaluation.

## Scope exclusions

This audit performs no:

- reference-output access;
- checkpoint loading;
- model training or evaluation;
- calibration or final-world access;
- CUDA execution;
- candidate selection;
- architecture freeze;
- automatic merge;
- manual merge.

It is repository state, continuity, and handoff control only.
