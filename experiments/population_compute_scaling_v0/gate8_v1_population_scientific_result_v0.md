# Gate-8 v1 population scientific result

## Status

`G8_V1_POPULATION_SCIENTIFIC_EVALUATION_COMPLETE`

Frozen population-scaling outcome:

`G8_POSITIVE_CAPABILITY_SCALING`

Execution head:

`5f9980056da6c841020fd34e19b9c041d3c1c815`

The 1B Gemma reference was not loaded or executed in this phase. The joint population-versus-reference comparison remains unclassified.

## Exact external evidence

```text
population summary
6d30d773f11c1155df3346128385da9231610ea05e95937e5acccb5529fca3fe

225,792-row per-world ledger
45e36bda230440d4fa2342183154b474473498df51917e147a37e0baa81c3323

seed-0 transition-table artifact
ed616803c0d93880fd587670ec6c2b165b0be6b6f7c988d73a4b7682a727399d

seed-1 transition-table artifact
688c35725c03104d7ed95a4666bbb3c22af060bc49be10db51860616d48083b5

seed-2 transition-table artifact
f1d5f163cb0ea2d46eb5430e3096870e4e423d054247a7fc517f5f1db176ea07

source manifest
8214aa82733a4fab9148a3ea210fd110b0a85f857483f14b15cb53d0f451255d
```

The raw 97 MB per-world ledger and compiled transition-table artifacts remain external evidence identified by SHA-256. The byte-identical CRLF source manifest is committed.

## Scientific matrix

```text
conditions             21
worlds per condition   512
unique test worlds     10,752
checkpoint seeds       0, 1, 2
modes                   7
raw rows                225,792
bootstrap replicates   20,000
learned parameters     19,649
```

The full organism obtained accuracy `1.0` with bootstrap interval `[1.0, 1.0]` in every condition.

## Capability frontiers

```text
population    maximum solved depth
32            4
64            8
128           16
256           32
512           64
1024          128
```

The solved frontier is nondecreasing, rises strictly at all five population transitions, and the final depth is 32 times the initial depth. This exceeds the frozen requirement of at least three strict rises and at least fourfold final-to-initial growth.

## Causal guards

At `(512, 64)`:

```text
full accuracy                    1.000000
no-communication accuracy        0.000000
shuffled-worker accuracy         0.06640625
full - no communication CI       [1.000000, 1.000000]
full - shuffled worker CI        [0.91015625, 0.955078125]
```

At `(1024, 128)`:

```text
full accuracy                    1.000000
no-communication accuracy        0.000000
shuffled-worker accuracy         0.05859375
full - no communication CI       [1.000000, 1.000000]
full - shuffled worker CI        [0.919921875, 0.9609375]
```

Both preregistered causal guards exceed the frozen `0.20` minimum lower-bound delta.

## Transition-function finding

The three distinct admitted checkpoint files compile to one identical 2,048-entry function:

`4a29df3f45b2f24c7a180a9147dce396206040200dd80ceb6d4e03ba130d563c`

Independent audit verified every one of the `256 inbox codes × 8 transforms` entries. Each compiled transition exactly advances the carrier nibble modulo 16 and applies the frozen primitive transform to the symbol nibble. Consequently, each independently trained checkpoint learned the complete local transition algebra, and the shared worker applies that learned function recurrently over previously unseen larger graphs and depths.

This is a positive result for the preregistered fixed-parameter population-computation hypothesis. It is not evidence of broad intelligence or an unrestricted scaling law; it demonstrates exact compositional extrapolation on this frozen distributed-transformation benchmark.

## Independent audit

The permanent audit reproduced:

- all eight source-manifest identities and the outer manifest hash;
- the exact Git-head, empty-status, and run-config hashes;
- all 225,792 rows, their single schema, exact ordering, and absence of extras;
- all 10,752 deterministic world IDs and symbolic-oracle answers;
- every organism/control prediction and every resource count from the frozen runtime;
- all 147 condition metric rows and correctness-matrix hashes;
- all 147 deterministic paired bootstrap intervals;
- both causal-ablation rows and paired delta intervals;
- all six frontiers and the unchanged population-scaling classifier.

## Closed boundaries

```text
training performed                    false
reference model loaded                false
reference inference performed         false
joint reference comparison classified false
```

The next scientific boundary is a separate, guarded Gemma reference execution on the identical 10,752-world matrix. This population result must remain immutable while that reference phase is implemented and executed.
