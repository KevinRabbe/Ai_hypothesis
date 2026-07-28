# Gate-1 v1 external archive verification

Status: **VERIFIED EXTERNAL EVIDENCE OBJECT**

This record identifies and independently checks the externally preserved ZIP containing the first canonical Gate-1 v1 target-GPU result.

## Archive identity

- archive name: `gate1_resource_frontier_v1_first_target_gpu.zip`
- archive SHA-256: `00857172ba87896558b89037b18952deda1a0b7e8d42b8a49c857f7773d6f434`
- archive member count: `29`
- canonical measurement Git head: `18b201e22ca0a33feb4644c8ed8a09375e8e23ea`
- packaging Git head recorded by the repair: `bf4bca0a2e12c86aa4e9a453a451b841edf2fc49`

The archive bytes are external evidence and are not stored in this repository. This repository record pins their exact cryptographic identity.

## Bundle contents checked

The archive contains the raw measurement and provenance bundle, including:

- `relay_resource_frontier_v1.json`
- `relay_resource_frontier_v1.audit.json`
- `relay_resource_frontier_v1.report.md`
- `checkpoint-verification.json`
- `git-head.txt`
- zero-byte `git-status.txt`
- `nvidia-smi.txt`
- `packaging-repair.json`
- `result-manifest.sha256`
- `source-manifest.sha256`
- the frozen confirmation artifact, including the seed-1 checkpoint used by Gate 1

## Result-manifest verification

Every file named by `result-manifest.sha256` was present in the uploaded archive and recomputed byte-for-byte against its recorded SHA-256.

Result: **10 / 10 manifest entries matched**.

This includes the frozen seed-1 checkpoint, checkpoint verification, raw result JSON, independent audit JSON, report, measurement-head snapshot, reconstructed clean status snapshot, source manifest, packaging repair record, and NVIDIA environment capture.

## Independent result cross-check

The raw `relay_resource_frontier_v1.json` contains exactly `30` frozen comparisons.

Recomputing the descriptive speedup summaries directly from that raw JSON reproduced the independent auditor:

### Batch 1

- conditions: `15`
- admissible correctness cells: `15 / 15`
- parallel faster than compute-matched cached serial: `15 / 15`
- parallel faster than low-memory serial: `15 / 15`
- cached-serial / parallel geometric mean: `16.61860399646208x`
- low-memory-serial / parallel geometric mean: `20.82778782760734x`

### Batch 64

- conditions: `15`
- admissible correctness cells: `15 / 15`
- parallel faster than compute-matched cached serial: `15 / 15`
- parallel faster than low-memory serial: `15 / 15`
- cached-serial / parallel geometric mean: `15.418120575030793x`
- low-memory-serial / parallel geometric mean: `19.001010820941563x`

The independent audit inside the archive reports:

- `protocol_valid = true`
- `reasons = []`
- expected conditions `30`
- observed conditions `30`
- schedule-order rotations `[0, 1, 2]`

## Packaging-repair provenance

`packaging-repair.json` records:

- `benchmark_rerun = false`
- `timing_rerun = false`
- `audit_protocol_valid_before_packaging_repair = true`
- reconstructed Git status: `clean/empty`

The repair therefore changes packaging/provenance only and does not replace the first admitted timing result.

## Scientific-source continuity

Comparing measurement head `18b201e22ca0a33feb4644c8ed8a09375e8e23ea` to packaging head `bf4bca0a2e12c86aa4e9a453a451b841edf2fc49` shows changes only in packaging/CI/result-record surfaces:

- `.github/workflows/population-compute-gate-ci.yml`
- `experiments/population_compute_scaling_v0/gate1_v1_target_gpu_result.md`
- `scripts/finalize_gate1_resource_frontier_v1_existing.ps1`
- `scripts/run_gate1_resource_frontier.ps1`
- `scripts/run_gate1_resource_frontier_v1.ps1`

None of the scientific measurement implementation files pinned by `source-manifest.sha256` changed between measurement and packaging.

## Evidence status

The first Gate-1 v1 target-GPU timing result is therefore preserved as a cryptographically identified, internally manifest-consistent, independently audited external evidence object.

Any later timing run is a replication and must not replace this first admitted result.
