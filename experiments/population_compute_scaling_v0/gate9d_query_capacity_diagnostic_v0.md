# Gate-9D query-capacity diagnostic v0

## Status

**DEVELOPMENT-ONLY QUERY-PATH LOCALIZATION. NOT CONFIRMATION, NOT A NEW GATE-9
RESULT, AND NOT AUTHORITY TO OPEN LATER DIAGNOSTIC STAGES OR POPULATION
SCIENCE.**

The first fast diagnostic established:

```text
byte lookup                  247 / 247 in all seeds
Walsh parity linear          247 / 247 in all seeds
current query-only           90, 73, 70 / 247
current full-context         52, 58, 71 / 247
```

The two weakest output bits implement the highest-order parities in the fixed
operator: one five-input parity and one six-input parity. The frozen query-only
path is effectively an eight-bit input, one shared 24-unit tanh bottleneck, and
an eight-bit output head.

## Question

This diagnostic separates four remaining explanations:

1. **Training duration** — does the unchanged current query-only model pass
   after 4,096 steps when steps 1,025–4,096 continue at the frozen minimum
   learning rate?
2. **24-unit bottleneck** — does a direct raw-bit tanh model with 32 units pass?
3. **Higher raw-bit capacity** — if 32 fails, does 64 pass?
4. **Parity representation mismatch** — does the original 24-unit bottleneck
   pass when fed all 256 fixed Walsh parity features rather than eight raw bits?

## Variants

```text
current_query_only_1024    frozen worker query-only path, 1,024 steps
current_query_only_4096    same initialization and path, 4,096 steps
raw_bits_tanh_32           8 -> 32 tanh -> 8
raw_bits_tanh_64           8 -> 64 tanh -> 8
walsh_tanh_24              256 fixed parity features -> 24 tanh -> 8
```

Every variant uses the same 247 examples, three initialization seeds, AdamW
mechanics, gradient clipping, fixed-final evaluation, and exact thresholds.
The 4,096-step control follows the complete frozen schedule through step 1,024
and then holds the learning rate at `1e-4`.

## Ordered diagnoses

```text
G9D_QUERY_CAPACITY_FAILURE_NOT_REPRODUCED
G9D_QUERY_CAPACITY_TRAINING_BUDGET_LIMIT
G9D_QUERY_CAPACITY_TRAINING_BUDGET_MIXED
G9D_QUERY_CAPACITY_24_UNIT_BOTTLENECK
G9D_QUERY_CAPACITY_32_UNIT_MIXED
G9D_QUERY_CAPACITY_BETWEEN_32_AND_64
G9D_QUERY_CAPACITY_64_UNIT_MIXED
G9D_QUERY_CAPACITY_RAW_BIT_PARITY_MISMATCH
G9D_QUERY_CAPACITY_WALSH_MIXED
G9D_QUERY_CAPACITY_UNRESOLVED
```

The result also reconstructs each output bit's unique affine parity mask and
records per-bit accuracy at every checkpoint.

## Output

One command creates:

```text
aggregate-summary.json
curves.jsonl
final-runs.jsonl
predictions.jsonl
git-head.txt
git-status.txt
run-config.json
manifest.sha256
<output-root>.zip
```

No checkpoint is written. The output remains compact development evidence.

## Closed boundaries

The diagnostic contains no stage-2, stage-3, or stage-4 runtime, no Gate-9
scientific-world generator, no population runtime, no confirmation publisher,
no checkpoint selection, and no mutation path for any frozen result.
