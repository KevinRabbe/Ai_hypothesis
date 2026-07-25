"""Homogeneous worker-bank execution for Step 2 population experiments.

The bank enforces one architecture for every worker. Checkpoints may have different
learned weights, but mixed worker shapes are rejected before execution.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import NamedTuple, Sequence

import torch
from torch.func import functional_call, stack_module_state, vmap

from ai_hypothesis.step01.model import Step01Output, Step01Unit, UnitConfig


class PopulationOutput(NamedTuple):
    """Raw outputs for every worker and sample.

    Shapes:
    - label_logits: [workers, batch, labels]
    - uncertainty_logits: [workers, batch]
    """

    label_logits: torch.Tensor
    uncertainty_logits: torch.Tensor


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    path: str
    step: int | None
    validation_score: float | None
    checkpoint_id: str | None = None


class HomogeneousWorkerBank:
    """A population of independently weighted workers with one shared architecture.

    The default ``vmap`` backend evaluates worker state through batched functional
    calls. ``forward`` preserves the Step 2 same-input population experiment, while
    ``forward_selected`` executes one selected worker per sample so the runtime can
    batch different Work Items without changing worker architecture.
    """

    def __init__(
        self,
        *,
        template: Step01Unit,
        params: dict[str, torch.Tensor],
        buffers: dict[str, torch.Tensor],
        checkpoints: tuple[LoadedCheckpoint, ...],
        device: torch.device,
        execution_backend: str = "vmap",
    ) -> None:
        if execution_backend not in {"vmap", "loop"}:
            raise ValueError("execution_backend must be 'vmap' or 'loop'")
        if not checkpoints:
            raise ValueError("worker bank must contain at least one checkpoint")

        worker_counts = {value.shape[0] for value in params.values()}
        worker_counts.update(value.shape[0] for value in buffers.values())
        if worker_counts and worker_counts != {len(checkpoints)}:
            raise ValueError("stacked worker state has inconsistent population width")

        resolved_checkpoints: list[LoadedCheckpoint] = []
        for worker_index, checkpoint in enumerate(checkpoints):
            checkpoint_id = checkpoint.checkpoint_id
            if checkpoint_id is None:
                worker_state = {
                    name: value[worker_index]
                    for name, value in params.items()
                }
                worker_state.update(
                    {
                        name: value[worker_index]
                        for name, value in buffers.items()
                    }
                )
                checkpoint_id = _worker_state_id(worker_state, template.config)
            elif not checkpoint_id.strip():
                raise ValueError("checkpoint_id must be non-empty when supplied")
            resolved_checkpoints.append(
                replace(checkpoint, checkpoint_id=checkpoint_id)
            )

        checkpoint_ids = tuple(
            checkpoint.checkpoint_id for checkpoint in resolved_checkpoints
        )
        if any(checkpoint_id is None for checkpoint_id in checkpoint_ids):
            raise RuntimeError("worker checkpoint identity resolution failed")
        if len(set(checkpoint_ids)) != len(checkpoint_ids):
            raise ValueError(
                "duplicate worker weight identities are not allowed in one population"
            )

        self.template = template
        self.params = params
        self.buffers = buffers
        self.checkpoints = tuple(resolved_checkpoints)
        self.device = device
        self.execution_backend = execution_backend

    @property
    def population_width(self) -> int:
        return len(self.checkpoints)

    @property
    def unit_config(self) -> UnitConfig:
        return self.template.config

    @property
    def checkpoint_ids(self) -> tuple[str, ...]:
        identities = tuple(
            checkpoint.checkpoint_id for checkpoint in self.checkpoints
        )
        if any(identity is None for identity in identities):
            raise RuntimeError("worker bank contains unresolved checkpoint identity")
        return tuple(str(identity) for identity in identities)

    @classmethod
    def from_checkpoints(
        cls,
        checkpoint_paths: Sequence[str | Path],
        *,
        device: str | torch.device = "cpu",
        execution_backend: str = "vmap",
    ) -> "HomogeneousWorkerBank":
        paths = tuple(Path(path) for path in checkpoint_paths)
        if not paths:
            raise ValueError("checkpoint_paths must not be empty")

        resolved_device = torch.device(device)
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")

        models: list[Step01Unit] = []
        metadata: list[LoadedCheckpoint] = []
        expected_config: UnitConfig | None = None

        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"worker checkpoint not found: {path}")
            checkpoint = torch.load(path, map_location=resolved_device)
            config = UnitConfig(**checkpoint["unit_config"])
            config.validate()

            if expected_config is None:
                expected_config = config
            elif config != expected_config:
                raise ValueError(
                    "mixed worker architectures are not allowed in one population: "
                    f"{path} has {config}, expected {expected_config}"
                )

            model = Step01Unit(config).to(resolved_device)
            model.load_state_dict(checkpoint["model_state"], strict=True)
            model.eval()
            models.append(model)

            validation_metrics = checkpoint.get("validation_metrics") or {}
            validation_score = validation_metrics.get("macro_task_accuracy")
            metadata.append(
                LoadedCheckpoint(
                    path=str(path),
                    step=checkpoint.get("step"),
                    validation_score=(
                        float(validation_score)
                        if validation_score is not None
                        else None
                    ),
                    checkpoint_id=_worker_state_id(model.state_dict(), config),
                )
            )

        assert expected_config is not None
        params, buffers = stack_module_state(models)
        template = models[0]
        return cls(
            template=template,
            params=params,
            buffers=buffers,
            checkpoints=tuple(metadata),
            device=resolved_device,
            execution_backend=execution_backend,
        )

    def _validate_input(self, features: torch.Tensor, mask: torch.Tensor) -> None:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, sequence, feature]")
        if mask.ndim != 2:
            raise ValueError("mask must have shape [batch, sequence]")
        if features.shape[:2] != mask.shape:
            raise ValueError("feature and mask batch/sequence dimensions must match")

    def _functional_forward(
        self,
        params: dict[str, torch.Tensor],
        buffers: dict[str, torch.Tensor],
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> Step01Output:
        return functional_call(
            self.template,
            (params, buffers),
            (features, mask),
        )

    def _functional_single_forward(
        self,
        params: dict[str, torch.Tensor],
        buffers: dict[str, torch.Tensor],
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> Step01Output:
        output = self._functional_forward(
            params,
            buffers,
            features.unsqueeze(0),
            mask.unsqueeze(0),
        )
        return Step01Output(
            label_logits=output.label_logits.squeeze(0),
            uncertainty_logits=output.uncertainty_logits.squeeze(0),
        )

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> PopulationOutput:
        """Evaluate all workers against the same batch without mixing architectures."""

        self._validate_input(features, mask)
        features = features.to(self.device, non_blocking=True)
        mask = mask.to(self.device, non_blocking=True)
        self.template.eval()

        with torch.inference_mode():
            if self.execution_backend == "vmap":
                output = vmap(
                    self._functional_forward,
                    in_dims=(0, 0, None, None),
                    randomness="error",
                )(self.params, self.buffers, features, mask)
                return PopulationOutput(
                    label_logits=output.label_logits,
                    uncertainty_logits=output.uncertainty_logits,
                )

            worker_outputs: list[Step01Output] = []
            for worker_index in range(self.population_width):
                worker_params = {
                    name: value[worker_index] for name, value in self.params.items()
                }
                worker_buffers = {
                    name: value[worker_index] for name, value in self.buffers.items()
                }
                worker_outputs.append(
                    self._functional_forward(
                        worker_params,
                        worker_buffers,
                        features,
                        mask,
                    )
                )

            return PopulationOutput(
                label_logits=torch.stack(
                    [output.label_logits for output in worker_outputs], dim=0
                ),
                uncertainty_logits=torch.stack(
                    [output.uncertainty_logits for output in worker_outputs], dim=0
                ),
            )

    def forward_selected(
        self,
        worker_indices: Sequence[int] | torch.Tensor,
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> Step01Output:
        """Execute one selected worker for each sample in a heterogeneous-work batch."""

        self._validate_input(features, mask)
        indices = torch.as_tensor(worker_indices, dtype=torch.long, device=self.device)
        if indices.ndim != 1 or indices.shape[0] != features.shape[0]:
            raise ValueError("worker_indices must contain one index per batch sample")
        if bool(((indices < 0) | (indices >= self.population_width)).any()):
            raise IndexError("worker index is outside the population")

        features = features.to(self.device, non_blocking=True)
        mask = mask.to(self.device, non_blocking=True)
        selected_params = {
            name: value.index_select(0, indices) for name, value in self.params.items()
        }
        selected_buffers = {
            name: value.index_select(0, indices) for name, value in self.buffers.items()
        }
        self.template.eval()

        with torch.inference_mode():
            if self.execution_backend == "vmap":
                return vmap(
                    self._functional_single_forward,
                    in_dims=(0, 0, 0, 0),
                    randomness="error",
                )(selected_params, selected_buffers, features, mask)

            outputs: list[Step01Output] = []
            for sample_index, worker_index in enumerate(indices.tolist()):
                worker_params = {
                    name: value[worker_index] for name, value in self.params.items()
                }
                worker_buffers = {
                    name: value[worker_index] for name, value in self.buffers.items()
                }
                outputs.append(
                    self._functional_forward(
                        worker_params,
                        worker_buffers,
                        features[sample_index : sample_index + 1],
                        mask[sample_index : sample_index + 1],
                    )
                )

            return Step01Output(
                label_logits=torch.cat([output.label_logits for output in outputs], dim=0),
                uncertainty_logits=torch.cat(
                    [output.uncertainty_logits for output in outputs], dim=0
                ),
            )

    __call__ = forward


def _worker_state_id(
    state: Mapping[str, torch.Tensor],
    config: UnitConfig,
) -> str:
    """Hash exact learned state plus structural config into stable worker identity."""

    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            asdict(config),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("worker state must contain only tensors")
        if tensor.layout is not torch.strided:
            raise ValueError("worker state identity requires dense strided tensors")
        dense = tensor.detach().contiguous().reshape(-1)
        raw = dense.view(torch.uint8).cpu().numpy().tobytes()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return f"weights-sha256-{digest.hexdigest()}"
