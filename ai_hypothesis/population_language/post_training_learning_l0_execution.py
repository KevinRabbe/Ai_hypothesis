"""Checkpoint-independent execution primitives for Post-Training Learning L0."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import math
import os
import pathlib
import struct
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from . import l0_protocol
from . import post_training_learning_l0_adapter as adapter
from . import post_training_learning_l0_calibration as calibration
from . import post_training_learning_l0_protocol as protocol
from . import post_training_learning_l0_world as world
from .l0_reference_training import canonical_state_sha256

VERSION = "population-language-post-training-learning-l0-execution-primitives-v0"
BRANCH = "agent/population-language-post-training-learning-l0-execution-primitives-v0"
STATUS = "EXECUTION_PRIMITIVES_ONLY_NO_CHECKPOINT_OR_CALIBRATION_OR_FINAL_RESULT"
SOURCE_CALIBRATION_HEAD = "19aa701c475b19fc5b31409528948f21ad9fbdf4"

ARTIFACT_MAGIC = b"PTL0ADAPTERV0\0"
ARTIFACT_TENSOR_COUNT = len(adapter.NAMES)
ARTIFACT_DTYPE = torch.float32
ARTIFACT_DTYPE_LABEL = "torch.float32"
ARTIFACT_MAX_BYTES = protocol.MAX_PERSISTED_ADAPTATION_BYTES

SCHEDULE_SHA256_BY_UPDATES = {
    32: "391e3cedb1290c5956cd0d8b72fea240054f20914b64f81317966c70173ac81d",
    64: "0df432ac0bbfde71a84a199118d041467371b9c47f21c547d3aa06ebeced42ca",
    128: "77d166ee7e7fcb579acd16b3295ab56e9f42aed37ed4fa884fbca388461a7bed",
    256: "f8dbafd553ab4bca6d3d6b977a3cb8bc939adb18a86a46e9837d3c5bb9dd8958",
}


@dataclass(frozen=True)
class EncodedLearningExample:
    input_ids: Tensor
    target_id: int

    def validate(self) -> "EncodedLearningExample":
        if self.input_ids.dtype != torch.long or self.input_ids.ndim != 1:
            raise ValueError("encoded learning input must be rank-1 torch.long")
        if not 5 <= self.input_ids.numel() <= 8:
            raise ValueError("encoded learning prefix length drifted")
        if type(self.target_id) is not int:
            raise TypeError("encoded learning target must be an integer token ID")
        return self


@dataclass(frozen=True)
class ArtifactRecord:
    path: str
    sha256: str
    encoded_bytes: int
    raw_tensor_bytes: int
    rank: int


@dataclass(frozen=True)
class PairedBootstrapResult:
    observed_mean_gain: float
    ci95_lower: float
    resamples: int
    percentile: float
    seed: int
    procedure: str
    rng: str
    quantile_method: str


def canonical_base_state_sha256(model: torch.nn.Module) -> str:
    """Hash a base model using the existing reference-training canonical format."""
    return canonical_state_sha256(model)


def encode_learning_example(example: world.LearningExample) -> EncodedLearningExample:
    tokens = example.tokens
    if (
        len(tokens) < 7
        or tokens[0] != "<bos>"
        or tokens[1] != "<query>"
        or tokens[-3] != "<answer>"
        or tokens[-1] != "<eos>"
        or tuple(tokens[2:-4]) != example.operators
        or tokens[-4] != world.VALUE_TOKENS[example.input_value]
        or tokens[-2] != world.VALUE_TOKENS[example.output_value]
    ):
        raise ValueError("learning example token contract drifted")
    prefix = tokens[:-2]
    if prefix[-1] != "<answer>":
        raise RuntimeError("learning prefix must end at the answer marker")
    try:
        input_ids = torch.tensor(
            [l0_protocol.TOKEN_TO_ID[token] for token in prefix],
            dtype=torch.long,
        )
        target_id = l0_protocol.TOKEN_TO_ID[tokens[-2]]
    except KeyError as error:
        raise ValueError("learning example contains a token outside the base vocabulary") from error
    return EncodedLearningExample(input_ids, target_id).validate()


def adaptation_microbatch_ordinals(update_index: int) -> tuple[int, ...]:
    if (
        type(update_index) is not int
        or not 0 <= update_index < max(calibration.UPDATE_COUNTS)
    ):
        raise ValueError("adaptation update index lies outside the locked schedule")
    start = update_index * calibration.MICROBATCH_SIZE
    return tuple(
        (start + offset) % world.ADAPTATION_EXAMPLES
        for offset in range(calibration.MICROBATCH_SIZE)
    )


def adaptation_schedule_sha256(updates: int) -> str:
    if updates not in calibration.UPDATE_COUNTS:
        raise ValueError("update count lies outside the locked calibration grid")
    digest = hashlib.sha256()
    for update_index in range(updates):
        for ordinal in adaptation_microbatch_ordinals(update_index):
            digest.update(ordinal.to_bytes(2, "little"))
    value = digest.hexdigest()
    if value != SCHEDULE_SHA256_BY_UPDATES[updates]:
        raise RuntimeError("adaptation schedule hash drifted")
    return value


def build_locked_optimizer(
    adapted_model: adapter.BoundedPopulationAdapter,
    candidate: calibration.CalibrationCandidate,
) -> torch.optim.AdamW:
    candidate.validate()
    declared = adapted_model.declared_adaptation_parameters()
    if tuple(declared) != adapter.NAMES:
        raise ValueError("adapter parameter declaration drifted")
    if sum(value.numel() for value in declared.values()) != candidate.trainable_parameters:
        raise ValueError("candidate rank does not match the adapted model")
    if any(not value.requires_grad for value in declared.values()):
        raise ValueError("declared adaptation parameter is not trainable")
    if any(value.dtype != torch.float32 for value in declared.values()):
        raise ValueError("adaptation parameters must remain FP32")
    return torch.optim.AdamW(
        declared.values(),
        lr=candidate.learning_rate,
        betas=calibration.ADAMW_BETAS,
        eps=calibration.ADAMW_EPSILON,
        weight_decay=calibration.WEIGHT_DECAY,
    )


def _expected_shapes(rank: int) -> OrderedDict[str, tuple[int, ...]]:
    if rank not in adapter.SUPPORTED_RANKS:
        raise ValueError("artifact rank lies outside the adapter contract")
    return OrderedDict(
        (
            ("operator_embedding_delta", (8, 512)),
            ("encoder_down", (rank, 14_544)),
            ("encoder_up", (128, rank)),
            ("decoder_down", (rank, 128)),
            ("decoder_up", (14_544, rank)),
            ("value_logit_bias", (16,)),
        )
    )


def validate_adaptation_state(
    state: Mapping[str, Tensor],
) -> tuple[int, int]:
    if tuple(state) != adapter.NAMES:
        raise ValueError("adaptation state names or order drifted")
    encoder_down = state.get("encoder_down")
    if not isinstance(encoder_down, Tensor) or encoder_down.ndim != 2:
        raise ValueError("adapter rank cannot be inferred")
    rank = int(encoder_down.shape[0])
    shapes = _expected_shapes(rank)
    raw_bytes = 0
    for name, expected_shape in shapes.items():
        value = state[name]
        if not isinstance(value, Tensor):
            raise TypeError(f"adaptation state entry {name!r} is not a tensor")
        if tuple(value.shape) != expected_shape:
            raise ValueError(f"adaptation tensor shape drifted for {name}")
        if value.dtype != ARTIFACT_DTYPE:
            raise ValueError(f"adaptation tensor dtype drifted for {name}")
        if value.layout != torch.strided:
            raise ValueError(f"adaptation tensor layout drifted for {name}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"adaptation tensor contains non-finite values for {name}")
        raw_bytes += value.numel() * value.element_size()
    if raw_bytes != adapter.raw_fp32_bytes(rank):
        raise RuntimeError("adaptation raw-byte count drifted")
    if raw_bytes > protocol.MAX_PERSISTED_ADAPTATION_BYTES:
        raise ValueError("adaptation state exceeds the persisted-byte budget")
    return rank, raw_bytes


def encode_adaptation_artifact(state: Mapping[str, Tensor]) -> bytes:
    rank, raw_bytes = validate_adaptation_state(state)
    del rank, raw_bytes
    payload = bytearray(ARTIFACT_MAGIC)
    payload.extend(struct.pack("<I", ARTIFACT_TENSOR_COUNT))
    for name in adapter.NAMES:
        name_bytes = name.encode("utf-8")
        value = state[name].detach().cpu().contiguous()
        array = value.numpy().astype("<f4", copy=False)
        data = array.tobytes(order="C")
        payload.extend(struct.pack("<H", len(name_bytes)))
        payload.extend(name_bytes)
        payload.extend(struct.pack("<B", value.ndim))
        for dimension in value.shape:
            payload.extend(struct.pack("<I", int(dimension)))
        payload.extend(struct.pack("<Q", len(data)))
        payload.extend(data)
    encoded = bytes(payload)
    if len(encoded) > ARTIFACT_MAX_BYTES:
        raise ValueError("encoded adaptation artifact exceeds one MiB")
    return encoded


def _read_exact(view: memoryview, offset: int, size: int) -> tuple[memoryview, int]:
    end = offset + size
    if size < 0 or end > len(view):
        raise ValueError("adaptation artifact is truncated")
    return view[offset:end], end


def decode_adaptation_artifact(payload: bytes) -> OrderedDict[str, Tensor]:
    if not isinstance(payload, bytes):
        raise TypeError("adaptation artifact payload must be bytes")
    if len(payload) > ARTIFACT_MAX_BYTES:
        raise ValueError("adaptation artifact exceeds one MiB")
    view = memoryview(payload)
    magic, offset = _read_exact(view, 0, len(ARTIFACT_MAGIC))
    if bytes(magic) != ARTIFACT_MAGIC:
        raise ValueError("adaptation artifact magic drifted")
    raw_count, offset = _read_exact(view, offset, 4)
    (count,) = struct.unpack("<I", raw_count)
    if count != ARTIFACT_TENSOR_COUNT:
        raise ValueError("adaptation artifact tensor count drifted")

    state: OrderedDict[str, Tensor] = OrderedDict()
    for expected_name in adapter.NAMES:
        raw_name_length, offset = _read_exact(view, offset, 2)
        (name_length,) = struct.unpack("<H", raw_name_length)
        raw_name, offset = _read_exact(view, offset, name_length)
        try:
            name = bytes(raw_name).decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("adaptation artifact tensor name is not UTF-8") from error
        if name != expected_name:
            raise ValueError("adaptation artifact tensor names or order drifted")

        raw_ndim, offset = _read_exact(view, offset, 1)
        (ndim,) = struct.unpack("<B", raw_ndim)
        if not 1 <= ndim <= 4:
            raise ValueError("adaptation artifact tensor rank drifted")
        dimensions: list[int] = []
        for _ in range(ndim):
            raw_dimension, offset = _read_exact(view, offset, 4)
            (dimension,) = struct.unpack("<I", raw_dimension)
            if dimension <= 0:
                raise ValueError("adaptation artifact contains an empty dimension")
            dimensions.append(dimension)

        raw_data_length, offset = _read_exact(view, offset, 8)
        (data_length,) = struct.unpack("<Q", raw_data_length)
        expected_data_length = math.prod(dimensions) * 4
        if data_length != expected_data_length:
            raise ValueError("adaptation artifact tensor byte count drifted")
        raw_data, offset = _read_exact(view, offset, data_length)
        array = np.frombuffer(raw_data, dtype="<f4").copy().reshape(dimensions)
        state[name] = torch.from_numpy(array)

    if offset != len(view):
        raise ValueError("adaptation artifact contains trailing bytes")
    validate_adaptation_state(state)
    return state


def save_adaptation_artifact_create_once(
    path: pathlib.Path,
    state: Mapping[str, Tensor],
) -> ArtifactRecord:
    if not isinstance(path, pathlib.Path):
        raise TypeError("artifact path must be pathlib.Path")
    payload = encode_adaptation_artifact(state)
    rank, raw_bytes = validate_adaptation_state(state)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return ArtifactRecord(
        path=str(path),
        sha256=hashlib.sha256(payload).hexdigest(),
        encoded_bytes=len(payload),
        raw_tensor_bytes=raw_bytes,
        rank=rank,
    )


def load_adaptation_artifact(
    path: pathlib.Path,
    *,
    expected_sha256: str | None = None,
) -> OrderedDict[str, Tensor]:
    if not isinstance(path, pathlib.Path):
        raise TypeError("artifact path must be pathlib.Path")
    with path.open("rb") as handle:
        payload = handle.read(ARTIFACT_MAX_BYTES + 1)
    if len(payload) > ARTIFACT_MAX_BYTES:
        raise ValueError("adaptation artifact exceeds one MiB")
    observed_hash = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and observed_hash != expected_sha256:
        raise ValueError("adaptation artifact SHA-256 mismatch")
    return decode_adaptation_artifact(payload)


def _binary_correctness(values: Sequence[object], label: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{label} correctness must be a nonempty rank-1 sequence")
    if not bool(np.isin(array, (0, 1, False, True)).all()):
        raise ValueError(f"{label} correctness must contain only binary values")
    return array.astype(np.int8, copy=False)


def paired_bootstrap_lower_bound(
    baseline_correct: Sequence[object],
    adapted_correct: Sequence[object],
    *,
    world_seed: int,
    chunk_size: int = 64,
) -> PairedBootstrapResult:
    baseline = _binary_correctness(baseline_correct, "baseline")
    adapted = _binary_correctness(adapted_correct, "adapted")
    if baseline.shape != adapted.shape:
        raise ValueError("paired correctness vectors must have identical shape")
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("bootstrap chunk size must be a positive integer")

    seed = protocol.paired_bootstrap_seed(world_seed)
    differences = adapted.astype(np.float64) - baseline.astype(np.float64)
    gains = np.empty(protocol.PAIRED_BOOTSTRAP_RESAMPLES, dtype=np.float64)
    generator = np.random.Generator(np.random.PCG64(seed))
    episode_count = int(differences.size)
    for chunk_start in range(0, gains.size, chunk_size):
        chunk_end = min(chunk_start + chunk_size, gains.size)
        for resample_index in range(chunk_start, chunk_end):
            indices = generator.integers(
                0,
                episode_count,
                size=episode_count,
                endpoint=False,
            )
            gains[resample_index] = float(differences[indices].mean())

    lower = float(
        np.quantile(
            gains,
            protocol.PAIRED_BOOTSTRAP_LOWER_PERCENTILE,
            method=protocol.PAIRED_QUANTILE_METHOD,
        )
    )
    return PairedBootstrapResult(
        observed_mean_gain=float(differences.mean()),
        ci95_lower=lower,
        resamples=protocol.PAIRED_BOOTSTRAP_RESAMPLES,
        percentile=protocol.PAIRED_BOOTSTRAP_LOWER_PERCENTILE,
        seed=seed,
        procedure=protocol.PAIRED_PROCEDURE,
        rng=protocol.PAIRED_RNG,
        quantile_method=protocol.PAIRED_QUANTILE_METHOD,
    )


def validate_execution_primitives() -> dict[str, object]:
    checks = {
        "source_calibration_head_is_locked":
            len(SOURCE_CALIBRATION_HEAD) == 40
            and all(character in "0123456789abcdef" for character in SOURCE_CALIBRATION_HEAD),
        "artifact_payload_is_tensor_only":
            protocol.ARTIFACT_PAYLOAD_KIND
            == "DECLARED_TRAINABLE_TENSORS_ONLY",
        "artifact_tensor_names_are_exact":
            ARTIFACT_TENSOR_COUNT == 6 == len(adapter.NAMES),
        "artifact_budget_is_one_mib": ARTIFACT_MAX_BYTES == 1_048_576,
        "schedule_hashes_are_locked": all(
            adaptation_schedule_sha256(updates) == expected
            for updates, expected in SCHEDULE_SHA256_BY_UPDATES.items()
        ),
        "bootstrap_resamples_are_locked":
            protocol.PAIRED_BOOTSTRAP_RESAMPLES == 20_000,
        "bootstrap_rng_is_locked": protocol.PAIRED_RNG == "NUMPY_PCG64",
        "bootstrap_quantile_is_locked":
            protocol.PAIRED_QUANTILE_METHOD == "linear",
        "no_checkpoint_or_result_status":
            STATUS
            == "EXECUTION_PRIMITIVES_ONLY_NO_CHECKPOINT_OR_CALIBRATION_OR_FINAL_RESULT",
    }
    return {
        "status": STATUS,
        "version": VERSION,
        "source_calibration_head": SOURCE_CALIBRATION_HEAD,
        "artifact_magic_hex": ARTIFACT_MAGIC.hex(),
        "schedule_sha256_by_updates": dict(SCHEDULE_SHA256_BY_UPDATES),
        "checks": checks,
        "valid": all(checks.values()),
    }
