# Gate-9 contextual operator induction: three-seed final result v0

## Final classification

```text
G9_NOVEL_OPERATOR_INDUCTION_FAILED
```

All three ordered seeds completed the frozen training and validation protocol.
All three failed checkpoint admission. Because the protocol requires every
ordered seed checkpoint to pass before scientific local or graph evaluation,
Gate-9 v0 terminates at the training boundary.

No scientific test world was generated or executed.

## Per-seed results

```text
seed  exact accuracy       bit accuracy          full - query    full - shuffled
0     0.003326416015625    0.50048828125        -8 episodes     -3 episodes
1     0.004364013671875    0.5006904602050781     0 episodes     13 episodes
2     0.00390625           0.5026893615722656     8 episodes      6 episodes
```

Each seed remained near random byte accuracy and near 0.5 bit accuracy. The
public-support oracle remained exactly 1.0 for all 98,304 validation episodes.

## Aggregate

```text
validation episodes          98,304
full correct                 380
full accuracy                0.0038655598958333335
query-only correct           380
query-only accuracy          0.0038655598958333335
shuffled-context correct     364
shuffled-context accuracy    0.0037027994791666665
oracle correct               98,304
oracle accuracy              1.0
mean bit accuracy            0.5012893676757812
mean final training loss     0.6930264830589294
```

Across all three seeds, full-context performance equals query-only performance
exactly: 380 correct episodes each. Full context exceeds shuffled context by
only 16 of 98,304 episodes.

The frozen worker and training protocol therefore produced no evidence that the
learned model used the public support examples to infer unseen affine operators.

## Scope of the conclusion

This result establishes failure for the exact Gate-9 v0 combination:

- 19,649-parameter contextual worker;
- frozen nine-example support representation;
- frozen optimizer and learning-rate schedule;
- 262,144 training operators per seed;
- fixed final step-512 checkpoints;
- ordered initialization seeds 900900, 900901, and 900902.

It does not show that contextual operator induction is impossible in general.
It does not identify whether the limiting factor is architecture, supervision,
optimization, training duration, representation, or another frozen choice.

The positive Gate-8 population-computation result is not invalidated. Gate-9
tested a new prerequisite: induction of an unseen local operator from public
examples.

## Immutable identities

```text
final result Git blob        2818297e92355a45d2989a80ef412df54151816f
seed-0 result Git blob       b86141981eaedd5ad7f20c3e15a1d68f82a3d821
seed-1 result Git blob       c459d426584731565d2c609b4e74548e24c46372
seed-2 result Git blob       1e5cd44ddd94494d3f234f00a24012e7b24f4f31
```

## Closed boundaries

The final result slice performs no training, retraining, checkpoint selection,
operator generation, test-world generation, scientific execution, population
execution, or post-hoc rescue.

Future work must begin as a new diagnostic or Gate-9 v1 protocol. Gate-9 v0
scientific local and graph tests remain permanently closed.
