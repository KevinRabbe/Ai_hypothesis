"""Bound checkpoint identities and result schema for the Gate-7 scale-neutral transition bridge.

This module admits only the preregistered fresh low-scale depth-10 bridge. It does not expose a
high-scale Gate-7 population ladder, K search, checkpoint selection, training, or tuning surface.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .gate7_scale_neutral_model_prep import (
    GATE7_SCALE_NEUTRAL_PARAMETER_COUNT,
    Gate7ScaleNeutralModelConfig,
    Gate7ScaleNeutralScorer,
)
from .gate7_scale_neutral_transition_bridge_prep import Gate7ScaleNeutralGate6Adapter
from .gate7_scale_neutral_transition_training import GATE7_SCALE_NEUTRAL_TRANSITION_VERSION

GATE7_TRANSITION_TRAINING_GIT_HEAD = "07307650b2bbbfaa09b80e40caa4419ecdda2947"
GATE7_TRANSITION_BRIDGE_EXECUTION_ADMITTED = True
GATE7_TRANSITION_BRIDGE_HIGH_SCALE_OPENED = False
GATE7_TRANSITION_BRIDGE_EXPECTED = {
    0: {
        "sha256": "be1b18e22a1f713b48de8934aca1c0302f1255342684158c6ae31a7c3618a719",
        "fingerprint": "0be5c26d1056da1bf12a53be5ba5e6d1cadb7815eb117e2a77db269391c1c5aa",
        "training_seed": 0,
    },
    1: {
        "sha256": "a9b712736f440168ccd86ee070b8d301be0eb834b9d17418d40c0175207cadfb",
        "fingerprint": "b9685382992cb2f94454b6faa3675d458f236e7b0c7d8399bea256bedcb02e46",
        "training_seed": 1,
    },
    2: {
        "sha256": "cd7869511d07679cb1ad430743cad031299a575413e8a853ef9116d545c7480a",
        "fingerprint": "1bca7012c7350c3b3fe8c9790a65c36eea8fcd8bef3e123034817ff78673a7bb",
        "training_seed": 2,
    },
}


@dataclass(frozen=True, slots=True)
class Gate7TransitionCheckpointIdentity:
    checkpoint_index: int
    checkpoint_sha256: str
    parameter_fingerprint: str
    learned_parameter_count: int
    training_seed: int
    transition_version: str
    training_git_head: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Gate7OriginalCheckpointIdentity:
    checkpoint_index: int
    checkpoint_sha256: str
    parameter_fingerprint: str
    learned_parameter_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_verified_gate7_transition_checkpoint(
    *, checkpoint_index: int, checkpoint_path: Path, device: torch.device | str
) -> tuple[Gate7ScaleNeutralGate6Adapter, Gate7TransitionCheckpointIdentity]:
    """Load exactly one bound transition checkpoint and wrap it for the frozen Gate-6 scheduler."""

    if checkpoint_index not in GATE7_TRANSITION_BRIDGE_EXPECTED:
        raise ValueError("transition checkpoint index must be 0, 1 or 2")
    path = checkpoint_path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Gate-7 transition checkpoint does not exist: {path}")

    expected = GATE7_TRANSITION_BRIDGE_EXPECTED[checkpoint_index]
    observed_sha = sha256_file(path).lower()
    if observed_sha != expected["sha256"]:
        raise RuntimeError(
            f"transition checkpoint {checkpoint_index} SHA256 mismatch: "
            f"{observed_sha} != {expected['sha256']}"
        )

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"transition checkpoint {checkpoint_index} payload must be one mapping")
    if payload.get("transition_version") != GATE7_SCALE_NEUTRAL_TRANSITION_VERSION:
        raise RuntimeError(f"transition checkpoint {checkpoint_index} version mismatch")
    if payload.get("scientific_status") != "GATE7_SCALE_NEUTRAL_TRANSITION_CHECKPOINT_UNBRIDGED":
        raise RuntimeError(f"transition checkpoint {checkpoint_index} status is not the frozen unbridged state")
    if payload.get("bridge_opened") is not False or payload.get("gate7_high_scale_opened") is not False:
        raise RuntimeError(f"transition checkpoint {checkpoint_index} was produced after a forbidden opening")
    if payload.get("training_seed") != expected["training_seed"]:
        raise RuntimeError(f"transition checkpoint {checkpoint_index} training-seed mismatch")
    if payload.get("learned_parameter_count") != GATE7_SCALE_NEUTRAL_PARAMETER_COUNT:
        raise RuntimeError(f"transition checkpoint {checkpoint_index} parameter-count mismatch")
    if payload.get("parameter_fingerprint") != expected["fingerprint"]:
        raise RuntimeError(f"transition checkpoint {checkpoint_index} stored fingerprint mismatch")
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"transition checkpoint {checkpoint_index} state_dict is missing")

    scorer = Gate7ScaleNeutralScorer(Gate7ScaleNeutralModelConfig())
    scorer.load_state_dict(state_dict, strict=True)
    if scorer.trainable_parameter_count() != GATE7_SCALE_NEUTRAL_PARAMETER_COUNT:
        raise RuntimeError(f"transition checkpoint {checkpoint_index} reconstructed parameter-count mismatch")
    fingerprint = scorer.parameter_fingerprint()
    if fingerprint != expected["fingerprint"]:
        raise RuntimeError(f"transition checkpoint {checkpoint_index} reconstructed fingerprint mismatch")
    scorer.eval()
    scorer.to(device)
    adapter = Gate7ScaleNeutralGate6Adapter(scorer)
    adapter.eval()
    adapter.to(device)

    return adapter, Gate7TransitionCheckpointIdentity(
        checkpoint_index=checkpoint_index,
        checkpoint_sha256=observed_sha,
        parameter_fingerprint=fingerprint,
        learned_parameter_count=scorer.trainable_parameter_count(),
        training_seed=int(expected["training_seed"]),
        transition_version=GATE7_SCALE_NEUTRAL_TRANSITION_VERSION,
        training_git_head=GATE7_TRANSITION_TRAINING_GIT_HEAD,
    )
