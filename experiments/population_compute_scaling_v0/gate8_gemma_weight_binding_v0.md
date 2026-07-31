# Gate-8 exact Gemma model-file binding v0

## Status

**PRE-EXPOSURE DOWNLOAD-AND-HASH EXECUTION PATH — MODEL INSTANTIATION,
TOKENIZER LOADING, TRAINING, INFERENCE AND SCIENTIFIC-TEST WORLD GENERATION
REMAIN CLOSED.**

Base: exact qualified three-seed scientific-evaluation protocol head:

```text
6bb89111a47713bea0a23bb1cae662ed5ec56b42
```

This is the final prerequisite before scientific-test exposure. It binds the
exact frozen Gemma model bytes without constructing a model object or opening a
benchmark namespace.

## Frozen remote identity

```text
repository  google/gemma-3-1b-it
revision    dcc83ea841ab6100d6b47a070329e1ba4cf78752
```

Gemma license acceptance and authenticated repository access are required. The
PowerShell wrapper refuses to run without explicit access attestation.

## Exact download allowlist

Only these three files may become visible in the local model snapshot:

```text
config.json
generation_config.json
model.safetensors
```

The runner downloads each file independently with `hf_hub_download` at the
exact revision. It does not use a broad repository snapshot, model loader,
tokenizer loader or framework-specific deserializer.

The already-qualified tokenizer binding freezes the same `config.json` as:

```text
19cb5d28c97778271ba2b3c3df47bf76bdd6706724777a2318b3522230afe91e
```

The model-file binding must reproduce that SHA-256 exactly. This prevents the
model and tokenizer stages from silently binding different repository content.

Qualified tokenizer evidence remains bound by:

```text
tokenizer result
c8d6adb733cadbbd251d91d35f9d224e255705dac49ba144655717f9f4ab7b8d

tokenizer manifest
21de192eb57c0759fbf2236fae2252e5319696b71689ada1471b74a9f1315a88
```

## Config semantic guards

In addition to the exact file hash, the standard-library parser verifies:

```text
architecture             Gemma3ForCausalLM
model type               gemma3_text
source dtype             bfloat16
hidden size              1152
intermediate size        6912
hidden layers            26
attention heads          4
key/value heads          1
maximum positions        32768
vocabulary size          262144
```

These checks are diagnostic redundancy. The exact SHA-256 remains the primary
immutable identity.

`generation_config.json` is parsed and bound, including its token identifiers
and source sampling settings. It does not control the scientific run. The
scientific protocol independently requires greedy decoding at temperature zero
with at most 64 generated tokens.

## Safetensors inspection without tensor loading

The binding reads only:

1. the first eight little-endian bytes containing header length;
2. the UTF-8 JSON Safetensors header;
3. operating-system file size metadata;
4. the complete file stream for SHA-256.

It never materializes tensor data as numerical arrays.

The header audit verifies:

- valid UTF-8 JSON with no duplicate keys;
- exact descriptor keys: dtype, shape and data offsets;
- supported dtype and non-negative integral shapes;
- shape-derived parameter count;
- dtype-derived storage size;
- contiguous, non-overlapping, gap-free data offsets beginning at zero;
- final data offset equal to the physical data section size;
- complete tensor, parameter, dtype and byte ledgers;
- string-only metadata;
- all weights exclusively BF16;
- total learned parameters inside the frozen 0.9–1.1 billion reference class.

The result preserves the complete tensor ledger. The approximately 2 GB
`model.safetensors` binary remains external and is identified only by its exact
SHA-256 in Git history after the run is independently audited.

## Output contract

A successful local execution writes:

```text
<output-root>/git-head.txt
<output-root>/git-status.txt
<output-root>/run-config.json
<output-root>/manifest.sha256
<output-root>/binding/gate8-gemma-weight-binding.json
<output-root>/binding/model-snapshot/config.json
<output-root>/binding/model-snapshot/generation_config.json
<output-root>/binding/model-snapshot/model.safetensors
```

The result records:

- exact repository and revision;
- `huggingface-hub` version;
- file SHA-256 and byte sizes;
- frozen config and generation-config semantics;
- complete Safetensors tensor/parameter/dtype/byte ledger;
- qualified tokenizer result identities;
- all closed-boundary flags.

The outer manifest hashes every output file except itself. A later result-only
slice commits only the compact result, source manifest and scientific record;
the model binary never enters Git.

## Closed boundaries

The binding must record:

```text
model files downloaded        true
model-file binding complete   true
model instantiated            false
tokenizer loaded              false
training performed            false
inference performed           false
scientific-test worlds        false
```

There is no Torch import, Transformers import, `from_pretrained`, model class,
tokenizer class, generation call, benchmark encoder or world generator in the
binding path.

## Qualification strategy

CI remains network-free and model-free. It performs:

- standard-library compilation and synthetic Safetensors tests;
- exact plan/config/file-matrix regressions;
- malformed header, gap, overlap, storage, dtype and duplicate-key failures;
- source guards against model/tokenizer/inference/test surfaces;
- Windows wrapper smoke before packages, network or file download;
- exact six-file branch-scope proof.

No CI job downloads Gemma or exposes scientific-test data.

## Local execution after qualification

The operator must use the exact qualified branch head, a clean working tree, an
authenticated Hugging Face account with accepted Gemma access, and a fresh
output directory.

```powershell
.\scripts\bind_gate8_gemma_weights.ps1 `
    -HuggingFaceLicenseAndAccessAttested `
    -OutputRoot "F:\gate8_gemma_weight_binding_v0"
```

The complete terminal output, compact result JSON and outer manifest must be
preserved. The large model file remains local.

## Next boundary

After the external result and model SHA-256 are audited and permanently
qualified, a separate guarded joint scientific-execution and independent-audit
slice may bind the three exact population checkpoints plus this exact Gemma
model identity. Only that later slice may generate the frozen scientific-test
worlds and perform inference.
