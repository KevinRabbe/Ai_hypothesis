# Gate-8 Gemma tokenizer binding v0

## Status

**EXACT TOKENIZER-ONLY BINDING ADMITTED — MODEL WEIGHTS, INFERENCE, TRAINING AND SCIENTIFIC TEST EXECUTION CLOSED.**

Exact qualified encoder head:

`9882256ae0152bc266dc4d96cab3bbeb0c4ef95b`

Frozen official reference:

```text
repository = google/gemma-3-1b-it
revision   = dcc83ea841ab6100d6b47a070329e1ba4cf78752
```

The revision is a full immutable Hugging Face repository commit. Mutable `main`, short commit IDs, mirrors and derivative repositories are forbidden.

## Access boundary

The official Gemma repository is license-gated. The binding wrapper requires explicit confirmation that the user has:

- accepted the Gemma usage terms on Hugging Face;
- configured an authorized Hugging Face token locally.

CI never accesses Hugging Face and never receives a token.

## Exact download allowlist

The binding runner uses the immutable revision and downloads only:

```text
added_tokens.json
config.json
special_tokens_map.json
tokenizer.json
tokenizer.model
tokenizer_config.json
```

The local snapshot must contain exactly those six visible files. Hub cache metadata is ignored by the scientific file-set validator.

The following remain forbidden:

- `model.safetensors`;
- PyTorch model binaries;
- GGUF files;
- checkpoints ending in `.pt` or `.pth`;
- generation configuration;
- any causal-language-model class.

Every tokenizer/config file receives an independent SHA-256 identity in the binding artifact.

## Exact token-count procedure

The tokenizer is loaded only from the verified local snapshot with:

```text
local_files_only = true
trust_remote_code = false
use_fast = true
```

For each of the 21 frozen contract prompts, the runner applies the tokenizer's own chat template to one user message with:

```text
add_generation_prompt = true
tokenize = true
return_dict = true
```

No second special-token pass is added.

The artifact records:

- exact repository and revision;
- six tokenizer/config file hashes;
- tokenizer class;
- exact `transformers`, `tokenizers` and `huggingface_hub` versions;
- chat-template SHA-256;
- canonical prompt SHA-256;
- ASCII bytes and exact input-token count for all 21 conditions;
- the maximum-token condition;
- the frozen 24,576-token limit.

Any condition above 24,576 tokens rejects the binding.

## World boundary

The binding run generates only:

- the eight frozen public demonstration worlds;
- one contract-namespace target world per frozen condition.

It generates no scientific-test world and consumes no scientific answer.

## Output state

A successful artifact must state:

```text
tokenizer_bound = true
model_bound = false
model_weights_downloaded = false
inference_performed = false
scientific_test_worlds_generated = false
```

The PowerShell wrapper scans the complete output root and rejects any model-weight file.

## Qualification boundary

CI uses synthetic tokenizer files and fake tokenizer objects. It validates:

- the exact official identity and full revision;
- the exact six-file allowlist;
- file hashing and missing/extra-file rejection;
- chat-template hashing and deterministic token extraction;
- the exact 21-condition matrix and hard token limit;
- tokenizer-bound/model-closed artifact semantics;
- tokenizer-only runner source;
- Windows wrapper smoke before packages, network or prompt construction.

CI downloads no tokenizer and performs no external access.

## Next boundary

After a successful user-side tokenizer binding, the exact result and hashes must be recorded before either:

1. binding the 1B model weights at the same immutable revision; or
2. admitting the fixed-parameter Gate-8 organism architecture and training protocol.

Neither model inference nor scientific comparison is admitted by this stage.
