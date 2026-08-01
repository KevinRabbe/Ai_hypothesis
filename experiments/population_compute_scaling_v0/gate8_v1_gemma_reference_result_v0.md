# Gate-8 v1 Gemma reference result

## Status

`G8_V1_GEMMA_REFERENCE_EVALUATION_COMPLETE`

Execution head:

`4ab5dd3856e7bdb5afefa2a92da4fef056102995`

The frozen Gemma 3 1B reference completed inference on the exact 10,752-world Gate-8 v1 test matrix. The population organism was not rerun, no training occurred, and the joint population-versus-reference classifier remains closed.

## Exact external evidence

```text
reference summary
7e7f8002b41d25d6448ecaea6882fa84926b006d95c1aa024a94b774b0b305ab

10,752-row reference ledger
dda1009295378f4626b444b016b7aed2ff06c3468dc8b385d64809d4704a4706

10,752-row prompt index
e238d801743939acaa362455410f85edd6a67d50f189d68a09bf75ffb63c60ab

transactional SQLite ledger
7173853a236b777a02596bce3b61abecef3e61d52df661eb39422325fbb224a1

source manifest
3fc0628c0c5fb56901160f35639c708a44cb2501540db9ea5022dce0e374b743
```

The raw JSONL ledgers and SQLite database remain external evidence identified by SHA-256. The byte-identical CRLF source manifest is committed.

## Frozen reference

```text
repository            google/gemma-3-1b-it
revision              dcc83ea841ab6100d6b47a070329e1ba4cf78752
parameters            999,885,952
weights               BF16
attention             SDPA
demonstrations         8
batch size             1
maximum input tokens   24,576
maximum new tokens     64
decoding               greedy, one beam
```

The run used Python 3.11.9, Torch 2.9.1+cu130, Transformers 4.57.6, Tokenizers 0.22.2, NumPy 2.3.5, Hugging Face Hub 0.36.2, and Safetensors 0.8.0 on an NVIDIA GeForce RTX 4060 Ti.

## Test matrix

```text
split                 test
seed                  0
conditions            21
world indices         0..511 per condition
reference rows        10,752
maximum input tokens  9,892
```

The complete prompt index was generated and hash-bound before model loading. Each row binds its frozen sequence, condition, world index, world ID, prompt hash, ASCII byte count, input-token count, and symbolic-oracle answer.

## Reference outcome

```text
strictly correct          37 / 10,752
pooled accuracy           0.003441220238095238
valid one-symbol outputs  580 / 10,752
valid parse rate          0.053943452380952384
correct among valid       37 / 580
conditional accuracy      0.06379310344827586
```

The strict contract accepts only one hexadecimal symbol after trimming. Explanations, punctuation, multiple symbols, empty output, and all other continuations are invalid and score incorrect.

The valid-output conditional accuracy is descriptively close to the `1/16` rate expected from uniform symbol guessing. This result record does not introduce a post-hoc chance classifier or alter the preregistered population-versus-reference comparison.

All six population-1,024 conditions produced zero valid outputs and zero correct answers. Their mean output lengths were approximately 62 tokens, near the frozen 64-token generation cap. This is recorded as observed reference behavior, not as a protocol change.

## Condition observations

The highest condition accuracy was `9/512 = 0.017578125` at `(population=256, depth=4)`. Conditions below population 256 and all population-1,024 conditions scored zero. The complete 21-row metric table, correctness-vector hashes, deterministic 20,000-sample bootstrap intervals, token counts, wall times, and peak-device allocations are preserved in the compact result JSON.

## Resource accounting

```text
total measured inference wall seconds  22,662.851745802443
maximum peak allocated device bytes     6,637,060,608
mean generated tokens per row           23.594494047619047
```

The outer wrapper elapsed approximately 22,939.5 seconds including prompt-index work, model setup, exports, and final hashing.

## Independent audit

The permanent audit reproduced:

- all 10,752 prompt rows and all 10,752 result rows in exact contiguous order;
- every prompt/result identity field and all unique world IDs and prompt hashes;
- SQLite integrity, contiguous primary keys, and byte-equivalent JSON payloads;
- all 21 condition metrics and all correctness-vector hashes;
- all 21 deterministic 20,000-sample bootstrap intervals;
- pooled accuracy, valid-parse rate, token means, wall-time means, and peak memory;
- all external artifact hashes, run-config identity, Git-head identity, empty status, and outer manifest hash.

## Closed boundaries

```text
training performed                       false
population execution performed           false
joint reference comparison classified    false
reference result record only              true
```

This immutable result does not classify the population-versus-reference comparison. The next scientific boundary is a separate, guarded finalizer that reads the already frozen population and Gemma ledgers, verifies their identities, computes the preregistered paired comparison, and emits the final Gate-8 v1 classification without loading Gemma or rerunning either system.
