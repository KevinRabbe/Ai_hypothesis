# Gate-8 Gemma tokenizer binding v0 — result

## Status

**VALID EXACT TOKENIZER-BINDING EVIDENCE — ALL 21 FROZEN REFERENCE PROMPTS FIT THE 24,576-TOKEN LIMIT.**

Exact admitted execution head:

`d3e817a8b3554ecb335626f4ca966fe702474503`

Exact qualified encoder head:

`9882256ae0152bc266dc4d96cab3bbeb0c4ef95b`

Frozen official reference:

```text
repository = google/gemma-3-1b-it
revision   = dcc83ea841ab6100d6b47a070329e1ba4cf78752
```

The successful binding performed no training, model binding, model-weight download, inference, or scientific-test world generation.

## Artifact identity

```text
result_sha256   = c8d6adb733cadbbd251d91d35f9d224e255705dac49ba144655717f9f4ab7b8d
manifest_sha256 = 21de192eb57c0759fbf2236fae2252e5319696b71689ada1471b74a9f1315a88
```

The exact source JSON and complete output manifest are preserved under:

```text
experiments/population_compute_scaling_v0/gate8_gemma_tokenizer_binding_result_v0/
```

## Bound tokenizer identity

```text
tokenizer_class       = GemmaTokenizerFast
transformers          = 4.57.6
tokenizers            = 0.22.2
huggingface_hub       = 0.36.2
chat_template_sha256  = 7de1c58e208eda46e9c7f86397df37ec49883aeece39fb961e0a6b24088dd3c4
```

Tokenizer/config file identities:

| File | SHA-256 |
| --- | --- |
| `added_tokens.json` | `50b2f405ba56a26d4913fd772089992252d7f942123cc0a034d96424221ba946` |
| `config.json` | `19cb5d28c97778271ba2b3c3df47bf76bdd6706724777a2318b3522230afe91e` |
| `special_tokens_map.json` | `2f7b0adf4fb469770bb1490e3e35df87b1dc578246c5e7e6fc76ecf33213a397` |
| `tokenizer.json` | `4667f2089529e8e7657cfb6d1c19910ae71ff5f28aa7ab2ff2763330affad795` |
| `tokenizer.model` | `1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c` |
| `tokenizer_config.json` | `bfe25c2735e395407beb78456ea9a6984a1f00d8c16fa04a8b75f2a614cf53e1` |

## Exact 21-condition token matrix

| Population | Depth | ASCII bytes | Input tokens |
| ---: | ---: | ---: | ---: |
| 32 | 4 | 2,951 | 2,801 |
| 64 | 4 | 3,207 | 3,057 |
| 64 | 8 | 3,207 | 3,057 |
| 128 | 4 | 3,720 | 3,570 |
| 128 | 8 | 3,720 | 3,570 |
| 128 | 16 | 3,721 | 3,571 |
| 256 | 4 | 4,744 | 4,594 |
| 256 | 8 | 4,744 | 4,594 |
| 256 | 16 | 4,745 | 4,595 |
| 256 | 32 | 4,745 | 4,595 |
| 512 | 4 | 6,792 | 6,439 |
| 512 | 8 | 6,792 | 6,420 |
| 512 | 16 | 6,793 | 6,443 |
| 512 | 32 | 6,793 | 6,412 |
| 512 | 64 | 6,793 | 6,428 |
| 1,024 | 4 | 10,889 | 9,789 |
| 1,024 | 8 | 10,889 | 9,796 |
| 1,024 | 16 | 10,890 | 9,811 |
| 1,024 | 32 | 10,890 | 9,781 |
| 1,024 | 64 | 10,890 | **9,843** |
| 1,024 | 128 | 10,891 | 9,820 |

The maximum exact input is 9,843 tokens at `(population=1024, depth=64)`. This is 40.05% of the frozen 24,576-token admission limit and leaves 14,733 tokens of headroom. The deepest condition `(1024,128)` also fits with 9,820 tokens.

## Frozen conclusion

The complete Gate-8 reference prompt matrix is compatible with the selected immutable Gemma 3 1B tokenizer and chat template. No condition requires truncation, graph omission, demonstration removal, prompt adaptation, or a larger context window.

This result establishes only tokenizer and input-budget admissibility. It does **not** establish:

- 1B model correctness on the task;
- model-weight identity;
- successful baseline inference;
- population-organism capability;
- capability scaling;
- population superiority or non-inferiority to the 1B reference.

## Next boundary

The next independent stages are:

1. admit and qualify the fixed 19,649-learned-parameter organism architecture and training path without scientific-test exposure;
2. bind the exact 1B model-weight files at the already frozen repository revision;
3. train the three frozen organism seeds and select checkpoints only under the preregistered development procedure;
4. admit joint scientific execution only after both systems and all immutable identities are qualified.
