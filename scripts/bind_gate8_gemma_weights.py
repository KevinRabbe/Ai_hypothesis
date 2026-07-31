#!/usr/bin/env python3
"""Download and bind the exact Gate-8 Gemma model files without loading them."""

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
BINDING_PATH = (
    REPO_ROOT
    / "ai_hypothesis/population_compute/gate8_gemma_weight_binding.py"
)


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load Gate8 module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"required package is not installed: {distribution}"
        ) from exc


def bind_gate8_gemma_weights(*, output_root: pathlib.Path) -> int:
    binding = _load(
        BINDING_PATH,
        "gate8_gemma_weight_binding_contract",
    )
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(
            f"Gate8 Gemma weight binding output already exists: {output_root}"
        )
    output_root.mkdir(parents=True)
    snapshot_root = output_root / "model-snapshot"
    snapshot_root.mkdir()

    hub_version = _package_version("huggingface-hub")
    from huggingface_hub import hf_hub_download

    token: bool | str = os.environ.get("HF_TOKEN") or True
    for filename in binding.GATE8_GEMMA_REQUIRED_MODEL_FILES:
        downloaded = pathlib.Path(
            hf_hub_download(
                repo_id=binding.GATE8_GEMMA_REPO_ID,
                filename=filename,
                revision=binding.GATE8_GEMMA_REVISION,
                local_dir=str(snapshot_root),
                token=token,
            )
        ).resolve()
        expected = (snapshot_root / filename).resolve()
        if downloaded != expected:
            raise RuntimeError(
                "Gate8 Gemma download returned an unexpected local path: "
                f"{downloaded}"
            )
        print(
            f"downloaded {filename} bytes={downloaded.stat().st_size}",
            flush=True,
        )

    file_hashes, file_sizes = binding.validate_gate8_gemma_model_snapshot(
        snapshot_root
    )
    config_semantics = binding.validate_gate8_gemma_config(
        snapshot_root / "config.json"
    )
    generation_semantics = binding.validate_gate8_gemma_generation_config(
        snapshot_root / "generation_config.json"
    )
    safetensors = binding.inspect_gate8_safetensors(
        snapshot_root / "model.safetensors"
    )
    safetensors.validate_gemma_weight_contract()

    summary = binding.Gate8GemmaWeightBindingSummary(
        repo_id=binding.GATE8_GEMMA_REPO_ID,
        revision=binding.GATE8_GEMMA_REVISION,
        huggingface_hub_version=hub_version,
        file_sha256=file_hashes,
        file_sizes=file_sizes,
        config_semantics=config_semantics,
        generation_config_semantics=generation_semantics,
        safetensors=safetensors,
    )
    summary.validate()

    payload = {
        "experiment_version": binding.GATE8_GEMMA_WEIGHT_BINDING_VERSION,
        "scientific_status": "GATE8_GEMMA_MODEL_FILE_BINDING_COMPLETE",
        "scientific_protocol_head": (
            binding.GATE8_GEMMA_WEIGHT_BINDING_PROTOCOL_HEAD
        ),
        "tokenizer_result_head": (
            binding.GATE8_GEMMA_WEIGHT_BINDING_TOKENIZER_RESULT_HEAD
        ),
        "model_binding": summary.to_dict(),
        "model_instantiated": False,
        "tokenizer_loaded": False,
        "training_performed": False,
        "inference_performed": False,
        "scientific_test_worlds_generated": False,
    }
    result_path = output_root / "gate8-gemma-weight-binding.json"
    _write_json(result_path, payload)

    print(
        json.dumps(
            {
                "status": "GATE8_GEMMA_MODEL_FILE_BINDING_COMPLETE",
                "result": str(result_path),
                "repo_id": summary.repo_id,
                "revision": summary.revision,
                "model_safetensors_sha256": file_hashes[
                    "model.safetensors"
                ],
                "tensor_count": safetensors.tensor_count,
                "parameter_count": safetensors.parameter_count,
                "dtype_parameter_counts": (
                    safetensors.dtype_parameter_counts
                ),
                "model_file_binding_complete": True,
                "model_instantiated": False,
                "inference_performed": False,
                "scientific_test_worlds_generated": False,
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
    return bind_gate8_gemma_weights(output_root=args.output_root)


if __name__ == "__main__":
    raise SystemExit(main())
