"""Strict reference-checkpoint contract for Post-Training Learning L0.

This module validates immutable Population Language L0 checkpoint bytes without
performing calibration, final-world evaluation, or any GPU work.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import io
import os
import pathlib
import stat
from typing import Mapping

import torch
from torch import Tensor, nn

from . import post_training_learning_l0_protocol as protocol
from . import post_training_learning_l0_world as world
from .l0_models import PopulationLanguageOrganism, count_parameters
from .l0_reference_training import (
    OPTIMIZER_STEPS,
    POPULATION_COMMUNICATION_ROUNDS,
    POPULATION_TOP_K,
    VERSION as REFERENCE_TRAINING_VERSION,
    canonical_state_sha256,
)

VERSION = "population-language-post-training-learning-l0-checkpoint-contract-v0"
BRANCH = "agent/population-language-post-training-learning-l0-checkpoint-contract-v0"
STATUS = "CHECKPOINT_CONTRACT_ONLY_NO_CHECKPOINT_DISCOVERY_OR_SCIENTIFIC_EXECUTION"
SOURCE_EXECUTION_PRIMITIVES_HEAD = "821449afe7381d4becc9c43dc456632b66b8f034"

REFERENCE_MODEL_NAME = "population"
REFERENCE_CHECKPOINT_PAYLOAD_KEYS = (
    "version",
    "model",
    "seed",
    "optimizer_step",
    "state_dict",
)
REFERENCE_CHECKPOINT_MAX_BYTES = 96 * 1024 * 1024
REFERENCE_RAW_STATE_BYTES = protocol.BASE_PARAMETER_COUNT * 4


@dataclass(frozen=True)
class LoadedReferenceCheckpoint:
    model: PopulationLanguageOrganism
    path: str
    seed: int
    file_bytes: int
    file_sha256: str
    canonical_state_sha256: str


def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _new_population_model() -> PopulationLanguageOrganism:
    # Construction must not perturb the caller's CPU RNG state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        return PopulationLanguageOrganism(
            communication_rounds=POPULATION_COMMUNICATION_ROUNDS,
            top_k=POPULATION_TOP_K,
        )


def _validate_expected_hashes(
    expected_file_sha256: str,
    expected_canonical_sha256: str,
) -> None:
    if not _is_sha256(expected_file_sha256):
        raise ValueError("expected checkpoint file SHA-256 is invalid")
    if not _is_sha256(expected_canonical_sha256):
        raise ValueError("expected canonical checkpoint SHA-256 is invalid")


def materialize_population_checkpoint_state(
    state: Mapping[str, Tensor],
    *,
    expected_canonical_sha256: str,
) -> PopulationLanguageOrganism:
    """Validate an exact production-shaped state dictionary and load it on CPU."""
    if not _is_sha256(expected_canonical_sha256):
        raise ValueError("expected canonical checkpoint SHA-256 is invalid")
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint state_dict must be a mapping")

    model = _new_population_model()
    template = model.state_dict()
    if tuple(state) != tuple(template):
        raise ValueError("checkpoint state tensor names or order drifted")

    raw_bytes = 0
    for name, expected in template.items():
        value = state[name]
        if not isinstance(value, Tensor):
            raise TypeError(f"checkpoint state entry {name!r} is not a tensor")
        if value.device.type != "cpu":
            raise ValueError(f"checkpoint tensor {name!r} is not on CPU")
        if value.layout != torch.strided:
            raise ValueError(f"checkpoint tensor {name!r} layout drifted")
        if tuple(value.shape) != tuple(expected.shape):
            raise ValueError(f"checkpoint tensor {name!r} shape drifted")
        if value.dtype != expected.dtype:
            raise ValueError(f"checkpoint tensor {name!r} dtype drifted")
        if not value.is_contiguous():
            raise ValueError(f"checkpoint tensor {name!r} is not contiguous")
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f"checkpoint tensor {name!r} contains non-finite values")
        raw_bytes += value.numel() * value.element_size()

    if raw_bytes != REFERENCE_RAW_STATE_BYTES:
        raise RuntimeError("checkpoint raw state byte count drifted")
    if count_parameters(model) != protocol.BASE_PARAMETER_COUNT:
        raise RuntimeError("fresh population base parameter count drifted")

    model.load_state_dict(OrderedDict((name, state[name]) for name in template), strict=True)
    observed = canonical_state_sha256(model)
    if observed != expected_canonical_sha256:
        raise ValueError("canonical checkpoint SHA-256 mismatch")
    model.eval()
    return model


def decode_reference_checkpoint(
    payload: bytes,
    *,
    expected_seed: int,
    expected_file_sha256: str,
    expected_canonical_sha256: str,
) -> tuple[PopulationLanguageOrganism, str]:
    """Decode and validate exact checkpoint bytes without path discovery."""
    _validate_expected_hashes(expected_file_sha256, expected_canonical_sha256)
    if expected_seed not in world.MODEL_INITIALIZATION_SEEDS:
        raise ValueError("expected model seed is outside the preregistered set")
    if not isinstance(payload, bytes):
        raise TypeError("reference checkpoint payload must be bytes")
    if not payload:
        raise ValueError("reference checkpoint payload is empty")
    if len(payload) > REFERENCE_CHECKPOINT_MAX_BYTES:
        raise ValueError("reference checkpoint exceeds the bounded file size")

    observed_file_sha256 = hashlib.sha256(payload).hexdigest()
    if observed_file_sha256 != expected_file_sha256:
        raise ValueError("reference checkpoint file SHA-256 mismatch")

    try:
        decoded = torch.load(
            io.BytesIO(payload),
            map_location="cpu",
            weights_only=True,
        )
    except Exception as error:
        raise ValueError("reference checkpoint could not be decoded safely") from error

    if type(decoded) is not dict:
        raise TypeError("reference checkpoint payload must decode to a plain dict")
    if tuple(decoded) != REFERENCE_CHECKPOINT_PAYLOAD_KEYS:
        raise ValueError("reference checkpoint payload keys or order drifted")
    if decoded["version"] != REFERENCE_TRAINING_VERSION:
        raise ValueError("reference checkpoint training version drifted")
    if decoded["model"] != REFERENCE_MODEL_NAME:
        raise ValueError("reference checkpoint model identity drifted")
    if type(decoded["seed"]) is not int or decoded["seed"] != expected_seed:
        raise ValueError("reference checkpoint model seed drifted")
    if (
        type(decoded["optimizer_step"]) is not int
        or decoded["optimizer_step"] != OPTIMIZER_STEPS
    ):
        raise ValueError("reference checkpoint optimizer step drifted")

    model = materialize_population_checkpoint_state(
        decoded["state_dict"],
        expected_canonical_sha256=expected_canonical_sha256,
    )
    return model, observed_file_sha256


def load_reference_checkpoint(
    path: pathlib.Path,
    *,
    expected_seed: int,
    expected_file_sha256: str,
    expected_canonical_sha256: str,
) -> LoadedReferenceCheckpoint:
    """Load one explicitly named, hash-pinned regular checkpoint file."""
    _validate_expected_hashes(expected_file_sha256, expected_canonical_sha256)
    if not isinstance(path, pathlib.Path):
        raise TypeError("reference checkpoint path must be pathlib.Path")
    if path.is_symlink():
        raise ValueError("reference checkpoint path must not be a symbolic link")

    try:
        metadata = path.stat()
    except OSError as error:
        raise ValueError("reference checkpoint path is unavailable") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("reference checkpoint path is not a regular file")
    if not 0 < metadata.st_size <= REFERENCE_CHECKPOINT_MAX_BYTES:
        raise ValueError("reference checkpoint file size is outside the contract")

    try:
        with path.open("rb") as handle:
            opened_metadata = os.fstat(handle.fileno())
            if not stat.S_ISREG(opened_metadata.st_mode):
                raise ValueError("opened reference checkpoint is not a regular file")
            if opened_metadata.st_size != metadata.st_size:
                raise ValueError("reference checkpoint changed before reading")
            payload = handle.read(REFERENCE_CHECKPOINT_MAX_BYTES + 1)
            if handle.read(1):
                raise ValueError("reference checkpoint contains unread trailing data")
    except OSError as error:
        raise ValueError("reference checkpoint could not be read") from error

    if len(payload) != metadata.st_size:
        raise ValueError("reference checkpoint changed while reading")
    model, observed_file_sha256 = decode_reference_checkpoint(
        payload,
        expected_seed=expected_seed,
        expected_file_sha256=expected_file_sha256,
        expected_canonical_sha256=expected_canonical_sha256,
    )
    return LoadedReferenceCheckpoint(
        model=model,
        path=str(path),
        seed=expected_seed,
        file_bytes=len(payload),
        file_sha256=observed_file_sha256,
        canonical_state_sha256=expected_canonical_sha256,
    )


def validate_checkpoint_contract() -> dict[str, object]:
    model = _new_population_model()
    state = model.state_dict()
    raw_bytes = sum(value.numel() * value.element_size() for value in state.values())
    checks = {
        "payload_keys_are_exact": REFERENCE_CHECKPOINT_PAYLOAD_KEYS
        == ("version", "model", "seed", "optimizer_step", "state_dict"),
        "checkpoint_size_is_bounded": REFERENCE_CHECKPOINT_MAX_BYTES == 96 * 1024 * 1024,
        "raw_state_bytes_are_exact": raw_bytes == REFERENCE_RAW_STATE_BYTES,
        "raw_state_fits_checkpoint_bound": raw_bytes < REFERENCE_CHECKPOINT_MAX_BYTES,
        "population_parameter_count_is_exact": count_parameters(model)
        == protocol.BASE_PARAMETER_COUNT,
        "reference_optimizer_step_is_exact": OPTIMIZER_STEPS == 4_096,
        "all_reference_state_tensors_are_cpu": all(
            value.device.type == "cpu" for value in state.values()
        ),
        "all_reference_state_tensors_are_fp32": all(
            value.dtype == torch.float32 for value in state.values()
        ),
        "model_seed_sets_match": tuple(world.MODEL_INITIALIZATION_SEEDS)
        == (120100, 120101, 120102),
    }
    return {
        "status": STATUS,
        "version": VERSION,
        "source_execution_primitives_head": SOURCE_EXECUTION_PRIMITIVES_HEAD,
        "reference_training_version": REFERENCE_TRAINING_VERSION,
        "reference_model": REFERENCE_MODEL_NAME,
        "reference_optimizer_steps": OPTIMIZER_STEPS,
        "reference_checkpoint_max_bytes": REFERENCE_CHECKPOINT_MAX_BYTES,
        "reference_raw_state_bytes": REFERENCE_RAW_STATE_BYTES,
        "payload_keys": list(REFERENCE_CHECKPOINT_PAYLOAD_KEYS),
        "checks": checks,
        "valid": all(checks.values()),
    }
