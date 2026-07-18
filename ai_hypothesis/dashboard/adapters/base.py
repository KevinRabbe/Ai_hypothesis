"""Adapter protocol for raw experiment artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..schemas import ExperimentGroupV1, ExperimentRunV1, IndexErrorV1


@dataclass(frozen=True, slots=True)
class ArtifactContext:
    path: Path
    artifact_id: str
    artifact_ref: str
    indexed_at: str


@dataclass(frozen=True, slots=True)
class AdapterMatch:
    matched: bool
    confidence: int = 0
    warning: str | None = None


@dataclass(slots=True)
class NormalizationResult:
    runs: list[ExperimentRunV1] = field(default_factory=list)
    groups: list[ExperimentGroupV1] = field(default_factory=list)
    errors: list[IndexErrorV1] = field(default_factory=list)


class ResultAdapter(Protocol):
    adapter_id: str

    def can_handle(self, payload: dict[str, Any], context: ArtifactContext) -> AdapterMatch:
        ...

    def normalize(
        self,
        payload: dict[str, Any],
        context: ArtifactContext,
    ) -> NormalizationResult:
        ...
