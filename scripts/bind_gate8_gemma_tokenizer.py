#!/usr/bin/env python3
"""Bind the exact Gate-8 Gemma tokenizer and count all contract prompts."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import pathlib
import sys
from typing import Any


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
WORLD_PATH = REPO_ROOT / "ai_hypothesis/population_compute/gate8_distributed_transformation_worlds.py"
ENCODER_PATH = REPO_ROOT / "ai_hypothesis/population_compute/gate8_distributed_transformation_encoder.py"
BINDING_PATH = REPO_ROOT / "ai_hypothesis/population_compute/gate8_gemma_tokenizer_binding.py"


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load Gate8 module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"required package is not installed: {distribution}") from exc


def _load_local_tokenizer(snapshot_root: pathlib.Path):
    transformers_version = _package_version("transformers")
    tokenizers_version = _package_version("tokenizers")
    hub_version = _package_version("huggingface-hub")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot_root),
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    return tokenizer, transformers_version, tokenizers_version, hub_version


def bind_gate8_gemma_tokenizer(*, output_root: pathlib.Path) -> int:
    worlds = _load(WORLD_PATH, "gate8_tokenizer_binding_worlds")
    encoder = _load(ENCODER_PATH, "gate8_tokenizer_binding_encoder")
    binding = _load(BINDING_PATH, "gate8_tokenizer_binding_contract")

    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Gate8 tokenizer binding output already exists: {output_root}")
    output_root.mkdir(parents=True)
    snapshot_root = output_root / "tokenizer-snapshot"

    from huggingface_hub import snapshot_download

    token: bool | str = os.environ.get("HF_TOKEN") or True
    downloaded = pathlib.Path(
        snapshot_download(
            repo_id=binding.GATE8_GEMMA_REPO_ID,
            revision=binding.GATE8_GEMMA_REVISION,
            local_dir=str(snapshot_root),
            allow_patterns=list(binding.GATE8_GEMMA_REQUIRED_TOKENIZER_FILES),
            token=token,
        )
    ).resolve()
    if downloaded != snapshot_root:
        raise RuntimeError("Gate8 tokenizer download returned an unexpected local directory")

    file_hashes = binding.validate_gate8_tokenizer_snapshot(snapshot_root)
    tokenizer, transformers_version, tokenizers_version, hub_version = (
        _load_local_tokenizer(snapshot_root)
    )
    chat_template_sha = binding.gate8_chat_template_sha256(tokenizer)

    demonstrations = tuple(
        worlds.generate_gate8_world(
            split="demonstration",
            seed=0,
            world_index=index,
            population=32,
            depth=4,
        )
        for index in range(8)
    )

    rows = []
    for population, depth in worlds.GATE8_VALID_CONDITIONS:
        target = worlds.generate_gate8_world(
            split="contract",
            seed=0,
            world_index=0,
            population=population,
            depth=depth,
        )
        prompt = encoder.encode_gate8_reference_prompt(target.public, demonstrations)
        prompt_budget = encoder.validate_gate8_reference_prompt_budget(prompt)
        token_ids = binding.gate8_prompt_token_ids(tokenizer, prompt)
        row = binding.Gate8TokenizerConditionCount(
            population=population,
            depth=depth,
            prompt_sha256=encoder.gate8_reference_prompt_sha256(prompt),
            ascii_bytes=prompt_budget.ascii_bytes,
            input_tokens=len(token_ids),
        )
        row.validate()
        rows.append(row)
        print(
            f"P={population:4d} D={depth:3d} "
            f"ASCII={row.ascii_bytes:5d} TOKENS={row.input_tokens:5d}",
            flush=True,
        )

    summary = binding.Gate8TokenizerBindingSummary(
        repo_id=binding.GATE8_GEMMA_REPO_ID,
        revision=binding.GATE8_GEMMA_REVISION,
        tokenizer_class=type(tokenizer).__name__,
        transformers_version=transformers_version,
        tokenizers_version=tokenizers_version,
        huggingface_hub_version=hub_version,
        chat_template_sha256=chat_template_sha,
        tokenizer_file_sha256=file_hashes,
        conditions=tuple(rows),
    )
    summary.validate()
    maximum = binding.gate8_maximum_token_condition(summary.conditions)
    payload = {
        "experiment_version": binding.GATE8_TOKENIZER_BINDING_VERSION,
        "scientific_status": "GATE8_GEMMA_TOKENIZER_BINDING_COMPLETE",
        "encoder_head": binding.GATE8_TOKENIZER_BINDING_ENCODER_HEAD,
        "tokenizer_binding": summary.to_dict(),
        "training_performed": False,
        "model_binding_performed": False,
        "model_weights_downloaded": False,
        "inference_performed": False,
        "scientific_test_worlds_generated": False,
    }
    result_path = output_root / "gate8-gemma-tokenizer-binding.json"
    _write_json(result_path, payload)
    print(
        json.dumps(
            {
                "status": "GATE8_GEMMA_TOKENIZER_BINDING_COMPLETE",
                "result": str(result_path),
                "repo_id": summary.repo_id,
                "revision": summary.revision,
                "maximum_population": maximum.population,
                "maximum_depth": maximum.depth,
                "maximum_input_tokens": maximum.input_tokens,
                "input_token_limit": binding.GATE8_GEMMA_MAX_INPUT_TOKENS,
                "tokenizer_bound": True,
                "model_bound": False,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    args = parser.parse_args()
    return bind_gate8_gemma_tokenizer(output_root=args.output_root)


if __name__ == "__main__":
    raise SystemExit(main())
