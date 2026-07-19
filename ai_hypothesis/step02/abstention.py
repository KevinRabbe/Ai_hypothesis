"""Two-stage abstention utilities for Step 2 diagnostics."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch


@dataclass(frozen=True, slots=True)
class AbstentionConfig:
    version: str
    candidate_source: str
    feature_names: tuple[str, ...]
    model_type: str
    threshold: float
    weights: tuple[float, ...] = ()
    bias: float = 0.0
    feature_mean: tuple[float, ...] = ()
    feature_std: tuple[float, ...] = ()

    def validate(self) -> None:
        if not self.version:
            raise ValueError("version must be non-empty")
        if not self.candidate_source:
            raise ValueError("candidate_source must be non-empty")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if self.model_type not in {"logistic_regression", "decision_stump", "fixed_score"}:
            raise ValueError("unsupported model_type")
        lengths = {len(self.feature_names)}
        if self.weights:
            lengths.add(len(self.weights))
        if self.feature_mean:
            lengths.add(len(self.feature_mean))
        if self.feature_std:
            lengths.add(len(self.feature_std))
        if len(lengths) != 1:
            raise ValueError("feature-dependent fields must have matching lengths")

    def to_json(self) -> str:
        self.validate()
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "AbstentionConfig":
        data = json.loads(payload)
        for key in ("feature_names", "weights", "feature_mean", "feature_std"):
            if key in data:
                data[key] = tuple(data[key])
        config = cls(**data)
        config.validate()
        return config

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "AbstentionConfig":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


def apply_abstention(
    candidate_labels: Sequence[str],
    abstain_scores: torch.Tensor,
    *,
    threshold: float,
) -> list[str]:
    """Return each candidate label or UNCERTAIN; never changes label A to label B."""

    if abstain_scores.ndim != 1:
        raise ValueError("abstain_scores must be one-dimensional")
    if len(candidate_labels) != abstain_scores.shape[0]:
        raise ValueError("candidate_labels length must match abstain_scores")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    return [
        "UNCERTAIN" if float(score) >= threshold else str(label)
        for label, score in zip(candidate_labels, abstain_scores)
    ]


def standardize_features(
    features: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError("features must be [samples, features]")
    if mean.shape != (features.shape[1],) or std.shape != (features.shape[1],):
        raise ValueError("mean/std shape must match feature dimension")
    return (features - mean) / std.clamp_min(1e-6)

_FORBIDDEN_FEATURE_TOKENS = ("truth", "oracle", "correct", "label_target", "ground_truth")


def validate_inference_feature_names(feature_names: Sequence[str]) -> None:
    """Reject feature names that imply ground-truth/oracle leakage."""

    for name in feature_names:
        lowered = name.lower()
        if any(token in lowered for token in _FORBIDDEN_FEATURE_TOKENS):
            raise ValueError(f"feature {name!r} appears to leak oracle/truth state")

