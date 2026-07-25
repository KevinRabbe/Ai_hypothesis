"""Homogeneous worker-bank execution for Step 2 population experiments.

The bank enforces one architecture for every worker. Checkpoints may have different
learned weights, but mixed worker shapes are rejected before execution.
"""

from __future__ import annotations

from dataclasses import dataclass
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


class HomogeneousWorkerBank:
    """A population of independently weighted workers with one shared architecture.

    The default ``vmap`` backend evaluates the same input batch across all workers
    using stacked model state. A deterministic loop backend is retained as a
    correctness/reference path and as a compatibility fallback for operations that
    cannot be vectorized on a particular PyTorch/device combination.
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

        self.template = template
        self.params = params
        self.buffers = buffers
        self.checkpoints = checkpoints
        self.device = device
        self.execution_backend = execution_backend

        worker_counts = {value.shape[0] for value in params.values()}
        worker_counts.update(value.shape[0] for value in buffers.values())
        if worker_counts and worker_counts != {len(checkpoints)}:
            raise ValueError("stacked worker state has inconsistent population width")

    @property
    def population_width(self) -> int:
        return len(self.checkpoints)

    @property
    def unit_config(self) -> UnitConfig:
        return self.template.config

    @property
    def selected_execution_backend(self) -> str:
        """Execution strategy used by ``forward_selected``."""

        return "grouped"

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
        if features.shape[1:] != (
            self.unit_config.sequence_length,
            self.unit_config.feature_width,
        ):
            raise ValueError("features do not match the homogeneous worker architecture")

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
        """Evaluate one selected worker per sample without all-worker expansion.

        Samples assigned to the same worker are grouped into one ordinary model
        batch. Only selected checkpoints execute; the entire population is never
        evaluated merely to obtain a few assigned worker results. A future fused
        implementation can replace this internal strategy without changing callers.
        """

        self._validate_input(features, mask)
        indices = torch.as_tensor(
            worker_indices,
            dtype=torch.long,
            device=self.device,
        )
        if indices.ndim != 1:
            raise ValueError("worker_indices must be one-dimensional")
        if indices.shape[0] != features.shape[0]:
            raise ValueError("worker_indices must contain one entry per sample")
        if bool((indices < 0).any()) or bool((indices >= self.population_width).any()):
            raise IndexError("worker index is outside the loaded population")

        features = features.to(self.device, non_blocking=True)
        mask = mask.to(self.device, non_blocking=True)
        self.template.eval()
        batch_size = features.shape[0]
        label_count = self.template.label_head.out_features
        label_logits = torch.empty(
            (batch_size, label_count),
            device=self.device,
            dtype=features.dtype,
        )
        uncertainty_logits = torch.empty(
            (batch_size,),
            device=self.device,
            dtype=features.dtype,
        )

        with torch.inference_mode():
            for worker_index in torch.unique(indices, sorted=True).tolist():
                sample_positions = torch.nonzero(
                    indices == worker_index,
                    as_tuple=False,
                ).squeeze(1)
                worker_params = {
                    name: value[worker_index] for name, value in self.params.items()
                }
                worker_buffers = {
                    name: value[worker_index] for name, value in self.buffers.items()
                }
                output = self._functional_forward(
                    worker_params,
                    worker_buffers,
                    features.index_select(0, sample_positions),
                    mask.index_select(0, sample_positions),
                )
                label_logits.index_copy_(
                    0,
                    sample_positions,
                    output.label_logits,
                )
                uncertainty_logits.index_copy_(
                    0,
                    sample_positions,
                    output.uncertainty_logits,
                )

        return Step01Output(
            label_logits=label_logits,
            uncertainty_logits=uncertainty_logits,
        )

    __call__ = forward
