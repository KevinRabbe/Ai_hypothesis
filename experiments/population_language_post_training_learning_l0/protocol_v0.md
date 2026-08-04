# Population Language Post-Training Learning L0 protocol v0

**Protocol only. No adaptation implementation, checkpoint result, or continual-learning claim is included.**

This protocol was fixed while the immutable Population Language L0 reference-training run was still in progress. It targets the exact reference implementation head `f7d997e828e2a8791592c060973080e3fe3c43bd` and the exact protocol base `bfd2111b65f805e6379ad45ecda6f5fe09d2a282`.

## Claim under test

A frozen 18,967,968-parameter population checkpoint can acquire a new operator system through a bounded learned neural artifact, preserve that improvement after a fresh-process restart, generalize from single operators to unseen four-operator compositions, and retain its original Population Language L0 capability.

The base checkpoint is immutable. A changed base-state hash invalidates the run.

## Worlds and splits

Eight reused nonce tokens denote deterministic world-specific affine permutations over 16 existing value tokens. The evaluator executes `f(x) = (a*x+b) mod 16` with odd `a`. The generator is an evaluation oracle and is unavailable to adaptation and model runtime.

Calibration worlds are `210100–210102`; final worlds are `220100–220102`. They are disjoint.

| Split | Count | Depth | World role |
|---|---:|---:|---|
| adaptation | 64 | 1 | calibration or final |
| direct holdout | 64 | 1 | calibration or final; disjoint inputs |
| calibration | 512 | 2 | calibration worlds only |
| validation | 2,048 | 3 | final worlds only |
| test | 8,192 | 4 | final worlds only |
| retention | 16,384 | original L0 | original test |

Complete deterministic fingerprints are mandatory. Calibration evidence includes adaptation, direct-holdout, and depth-2 fingerprints for all three calibration worlds. Final evidence includes adaptation, direct-holdout, validation, and test fingerprints for the paired final world.

## Bounded adaptation

- at most 189,680 trainable parameters;
- at most 1 MiB persisted artifact;
- exactly 64 adaptation examples;
- at most 256 optimizer updates;
- at most 4,096 example presentations;
- adapter initialization seed `700000 + model_seed`, independent of world identity;
- no raw adaptation examples or external retrieval during evaluation.

The persisted payload is `DECLARED_TRAINABLE_TENSORS_ONLY`. Lookup tables, inferred affine coefficients, serialized examples, executable rules, symbolic fitting, and alternate answer paths are invalid. Model logits are authoritative.

## Persistence

Each pair is evaluated before adaptation, immediately after adaptation, and after a fresh process reloads the untouched base checkpoint plus the persisted artifact. Immediate and restarted direct, composition, and retention accuracies may differ by at most 0.001.

## Paired inference

The depth-4 gain uses `DETERMINISTIC_PAIRED_BOOTSTRAP_PERCENTILE_V0` with NumPy PCG64, 20,000 paired resamples, lower percentile 0.025, quantile method `linear`, and seed `world_seed + 900000`.

## Success

A valid run supports persistent post-training learning only when:

- direct-holdout gain is positive for every seed;
- mean depth-4 test gain is at least 0.02;
- depth-4 gain and its paired 95% lower bound are positive for every seed;
- mean original-L0 retention drop is at most 0.005;
- per-seed retention drop is at most 0.01.

A valid run that misses a threshold concludes `DOES_NOT_SUPPORT_PERSISTENT_POST_TRAINING_LEARNING`. Invalid evidence concludes `INVALID_RUN_NO_POST_TRAINING_LEARNING_CONCLUSION` and is never relabeled as a negative scientific result.
