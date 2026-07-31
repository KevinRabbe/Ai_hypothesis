# Gate-8 exact Gemma model-file binding result v0

## Status

**EXACT GEMMA MODEL-FILE BINDING COMPLETE — MODEL INSTANTIATION, TOKENIZER LOADING, TRAINING, INFERENCE AND SCIENTIFIC-TEST WORLD GENERATION REMAIN CLOSED.**

Execution head:

`1390c70565204736ab77c725cd6f49e6a5876124`

Scientific-evaluation protocol head:

`6bb89111a47713bea0a23bb1cae662ed5ec56b42`

## External evidence

```text
raw result SHA-256
c554b66068b04ade24e77bb561fb7fff148fc3fd9a6316e011f710b0f320c10d

source manifest SHA-256
99ae54872115207c1e703a7c63fb66a1f4c145741958e63354bcf074310ae51c
```

The repository stores the byte-identical CRLF source manifest and a compact immutable summary. The approximately 2 GB model binary remains external evidence identified by SHA-256 and is not committed.

## Frozen remote identity

```text
repository  google/gemma-3-1b-it
revision    dcc83ea841ab6100d6b47a070329e1ba4cf78752
```

Exact files:

```text
config.json             899 bytes
SHA-256 19cb5d28c97778271ba2b3c3df47bf76bdd6706724777a2318b3522230afe91e

generation_config.json  215 bytes
SHA-256 fd9324becc53c4be610db39e13a613006f09fd6ef71a95fb6320dc33157490a3

model.safetensors       1,999,811,208 bytes
SHA-256 3d4ef8d71c14db7e448a09ebe891cfb6bf32c57a9b44499ae0d1c098e48516b6
```

The config hash exactly reproduces the earlier qualified tokenizer-stage config identity.

## Safetensors audit

The standard-library header inspection established:

```text
tensor count       340
parameter count    999,885,952
dtype              BF16 only
storage bytes      1,999,771,904
header bytes       39,296
complete file size 1,999,811,208
metadata format    pt
```

Independent audit of the compact result verified:

- 340 unique tensor names;
- 26 complete transformer layers, each with 13 tensors;
- only `model.embed_tokens.weight` and `model.norm.weight` outside the layer stack;
- every shape product equals its recorded parameter count;
- every BF16 tensor uses exactly two bytes per parameter;
- tensor offsets begin at zero, are contiguous and gap-free, and end at the exact data boundary;
- summed parameter and storage ledgers reproduce the recorded totals;
- header plus data bytes reproduce the exact model-file size.

No tensor payload was materialized and no model object was constructed.

## Model configuration

```text
architecture             Gemma3ForCausalLM
model type               gemma3_text
hidden width             1,152
intermediate width       6,912
layers                   26
attention heads          4
key/value heads          1
vocabulary               262,144
maximum positions        32,768
source dtype             bfloat16
```

## Closed scientific boundaries

```text
model instantiated              false
tokenizer loaded                false
training performed              false
inference performed             false
scientific-test worlds generated false
```

This result binds the final external prerequisite for the frozen 1B reference. It does not admit inference by itself.

## Next boundary

A separate guarded joint scientific-execution slice may now be implemented from this exact qualified result head. It must:

1. accept only the three exact 19,649-parameter checkpoints and this exact Gemma snapshot;
2. reproduce all artifact hashes before any test world is generated;
3. generate only the frozen shared `test`, seed `0`, indices `0..511` matrix;
4. execute all population modes and controls required by the scientific protocol;
5. run Gemma with the qualified tokenizer, BF16, greedy decoding, temperature zero and at most 64 new tokens;
6. preserve raw per-world evidence, exact oracle results, resource accounting and deterministic bootstrap inputs;
7. classify only after every required row is complete;
8. fail closed on any missing identity, condition, world, mode or reference result.
