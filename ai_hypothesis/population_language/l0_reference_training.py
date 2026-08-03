"""Exact reference-training and evaluation runner for Population Language L0."""
from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import math
import pathlib
import time
from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from . import l0_protocol as protocol
from .l0_data import LanguageBatch, materialize_batch
from .l0_models import (
    MatchedCausalTransformer,
    PopulationLanguageOrganism,
    count_parameters,
)

VERSION = "population-language-l0-reference-training-v0"
BRANCH = "agent/population-language-l0-reference-training-v0"
BASE_HEAD = "6a991a2dbf0a0ac3a157966a8898362192439626"
STATUS = "POPULATION_LANGUAGE_L0_REFERENCE_TRAINING"
PASS = "POPULATION_LANGUAGE_L0_REFERENCE_RUN_VALID"
INVALID = "POPULATION_LANGUAGE_L0_REFERENCE_RUN_INVALID"

OPTIMIZER_STEPS = 4_096
GLOBAL_BATCH_SIZE = 256
DEFAULT_MICROBATCH = 8
DEFAULT_EVALUATION_MICROBATCH = 8
WARMUP_STEPS = 205
PEAK_LEARNING_RATE = 3e-4
BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.1
GRADIENT_CLIP = 1.0
TRAINING_LOG_INTERVAL = 64
POPULATION_TRAIN_WORKERS = 32
POPULATION_COMMUNICATION_ROUNDS = 6
POPULATION_TOP_K = 4
EXPECTED_TARGET_TOKENS_PER_EPISODE = 27
ANSWER_TOKEN_COUNT = 5
SCALING_SUPPORTS = "SUPPORTS_FIXED_PARAMETER_POPULATION_SCALING"
SCALING_DOES_NOT_SUPPORT = "DOES_NOT_SUPPORT_FIXED_PARAMETER_POPULATION_SCALING"
SCALING_INVALID = "INVALID_REFERENCE_RUN_NO_SCALING_CONCLUSION"

_DATA_PERMUTATION_MULTIPLIER = 65_537
_DATA_EPOCH_SHIFT = 8_191
_DEFINITION_LEFT = slice(1, 7)
_DEFINITION_RIGHT = slice(7, 13)


@dataclass(frozen=True)
class TrainingContract:
    microbatch: int
    evaluation_microbatch: int
    accumulation_steps: int
    warmup_steps: int = WARMUP_STEPS
    optimizer_steps: int = OPTIMIZER_STEPS
    global_batch_size: int = GLOBAL_BATCH_SIZE


@dataclass(frozen=True)
class CachedSplit:
    input_ids: Tensor
    target_ids: Tensor
    loss_mask: Tensor
    answer_mask: Tensor

    def batch(
        self,
        ordinals: Iterable[int],
        *,
        device: torch.device,
    ) -> LanguageBatch:
        locked = tuple(ordinals)
        if not locked:
            raise ValueError("cached reference batch cannot be empty")
        index = torch.tensor(locked, dtype=torch.long, device=self.input_ids.device)
        size = self.input_ids.shape[0]
        if min(locked) < 0 or max(locked) >= size:
            raise ValueError("cached reference ordinal lies outside the split")

        def select(values: Tensor) -> Tensor:
            selected = values.index_select(0, index)
            return selected if selected.device == device else selected.to(device)

        return LanguageBatch(
            input_ids=select(self.input_ids),
            target_ids=select(self.target_ids),
            loss_mask=select(self.loss_mask),
            answer_mask=select(self.answer_mask),
            ordinals=locked,
        )


class ReferenceDatasetCache:
    """Materialize each deterministic split once, outside the optimization hot path."""

    def __init__(
        self,
        splits: dict[str, CachedSplit],
        build_seconds: float,
        resident_bytes: int,
    ) -> None:
        self.splits = splits
        self.build_seconds = build_seconds
        self.resident_bytes = resident_bytes

    @classmethod
    def build(cls, device: torch.device) -> "ReferenceDatasetCache":
        started = time.perf_counter()
        counts = {
            "train": protocol.REFERENCE_TRAIN_EPISODES,
            "validation": protocol.REFERENCE_VALIDATION_EPISODES,
            "test": protocol.REFERENCE_TEST_EPISODES,
        }
        splits: dict[str, CachedSplit] = {}
        for split, count in counts.items():
            batch = materialize_batch(split, range(count), device="cpu")
            tensors = tuple(
                value.to(device)
                for value in (
                    batch.input_ids,
                    batch.target_ids,
                    batch.loss_mask,
                    batch.answer_mask,
                )
            )
            splits[split] = CachedSplit(*tensors)
        resident_bytes = sum(
            tensor.numel() * tensor.element_size()
            for cached in splits.values()
            for tensor in (
                cached.input_ids,
                cached.target_ids,
                cached.loss_mask,
                cached.answer_mask,
            )
        )
        return cls(splits, time.perf_counter() - started, resident_bytes)

    def batch(
        self,
        split: protocol.Split,
        ordinals: Iterable[int],
        *,
        device: torch.device,
    ) -> LanguageBatch:
        cached = self.splits.get(split)
        if cached is None:
            raise ValueError(f"unknown cached split: {split}")
        return cached.batch(ordinals, device=device)


def validate_contract(
    microbatch: int = DEFAULT_MICROBATCH,
    evaluation_microbatch: int = DEFAULT_EVALUATION_MICROBATCH,
) -> TrainingContract:
    for value, label in (
        (microbatch, "microbatch"),
        (evaluation_microbatch, "evaluation microbatch"),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{label} must be a positive integer")
    if microbatch > GLOBAL_BATCH_SIZE or GLOBAL_BATCH_SIZE % microbatch:
        raise ValueError("microbatch must divide the locked global batch exactly")
    if evaluation_microbatch > protocol.REFERENCE_TEST_EPISODES:
        raise ValueError("evaluation microbatch exceeds the largest locked split")
    return TrainingContract(
        microbatch=microbatch,
        evaluation_microbatch=evaluation_microbatch,
        accumulation_steps=GLOBAL_BATCH_SIZE // microbatch,
    )


def learning_rate_for_step(step: int) -> float:
    if type(step) is not int or not 0 <= step < OPTIMIZER_STEPS:
        raise ValueError("optimizer step lies outside the locked schedule")
    update = step + 1
    if update <= WARMUP_STEPS:
        return PEAK_LEARNING_RATE * update / WARMUP_STEPS
    progress = (update - WARMUP_STEPS) / (OPTIMIZER_STEPS - WARMUP_STEPS)
    return PEAK_LEARNING_RATE * 0.5 * (1.0 + math.cos(math.pi * progress))


def training_ordinals(step: int) -> tuple[int, ...]:
    if type(step) is not int or not 0 <= step < OPTIMIZER_STEPS:
        raise ValueError("training step lies outside the locked schedule")
    start = step * GLOBAL_BATCH_SIZE
    ordinals: list[int] = []
    for global_index in range(start, start + GLOBAL_BATCH_SIZE):
        epoch, position = divmod(global_index, protocol.REFERENCE_TRAIN_EPISODES)
        ordinal = (
            position * _DATA_PERMUTATION_MULTIPLIER
            + epoch * _DATA_EPOCH_SHIFT
        ) % protocol.REFERENCE_TRAIN_EPISODES
        ordinals.append(ordinal)
    if len(set(ordinals)) != GLOBAL_BATCH_SIZE:
        raise RuntimeError("training schedule produced a duplicate within one global batch")
    return tuple(ordinals)


@functools.lru_cache(maxsize=1)
def training_schedule_sha256() -> str:
    digest = hashlib.sha256()
    for step in range(OPTIMIZER_STEPS):
        for ordinal in training_ordinals(step):
            digest.update(ordinal.to_bytes(8, "little"))
    return digest.hexdigest()


def swap_definition_order(input_ids: Tensor) -> Tensor:
    if input_ids.dtype != torch.long or input_ids.ndim != 2:
        raise ValueError("definition-order swap requires rank-2 torch.long input")
    if input_ids.shape[1] < _DEFINITION_RIGHT.stop:
        raise ValueError("definition-order swap input is too short")
    swapped = input_ids.clone()
    left = input_ids[:, _DEFINITION_LEFT].clone()
    right = input_ids[:, _DEFINITION_RIGHT].clone()
    swapped[:, _DEFINITION_LEFT] = right
    swapped[:, _DEFINITION_RIGHT] = left
    return swapped


def transformer_forward_flops_per_episode(
    config: protocol.TransformerConfig = protocol.TransformerConfig(),
    *,
    sequence_length: int = protocol.MAX_SEQUENCE_LENGTH - 1,
) -> int:
    if type(sequence_length) is not int or not 1 <= sequence_length < protocol.MAX_SEQUENCE_LENGTH:
        raise ValueError("transformer FLOP sequence length is outside the model contract")
    sequence = sequence_length
    width = config.d_model
    per_layer = (
        8 * sequence * width * width
        + 4 * sequence * sequence * width
        + 4 * sequence * width * config.feed_forward
    )
    output = 2 * sequence * width * config.vocab_size
    return config.layers * per_layer + output


def population_forward_flops_per_episode(
    worker_count: int,
    config: protocol.PopulationConfig = protocol.PopulationConfig(),
    *,
    communication_rounds: int = POPULATION_COMMUNICATION_ROUNDS,
    top_k: int = POPULATION_TOP_K,
    sequence_length: int = protocol.MAX_SEQUENCE_LENGTH - 1,
) -> int:
    if type(worker_count) is not int or worker_count <= 0:
        raise ValueError("worker count must be a positive integer")
    if communication_rounds <= 0 or top_k <= 0:
        raise ValueError("population communication settings must be positive")
    if type(sequence_length) is not int or not 1 <= sequence_length < protocol.MAX_SEQUENCE_LENGTH:
        raise ValueError("population FLOP sequence length is outside the model contract")
    sequence = sequence_length
    token = config.token_width
    worker = config.worker_width
    lexical = (
        2 * token * config.lexical_encoder_width
        + 2 * config.lexical_encoder_width * worker
        + 2 * worker * config.lexical_decoder_width
        + 2 * config.lexical_decoder_width * token
        + 2 * token * config.vocab_size
    )
    initializer = 2 * worker_count * worker * worker
    per_round = (
        4 * worker_count * worker * config.router_dim
        + 2 * worker_count * worker_count * config.router_dim
        + 2 * worker_count * worker * worker
        + 2 * worker_count * min(top_k, worker_count) * worker
        + 12 * worker_count * worker * worker
        + 4 * worker_count * worker * config.worker_feed_forward
    )
    return sequence * (lexical + initializer + communication_rounds * per_round)


def estimated_training_flops(model: str) -> int:
    if model == "transformer":
        forward = transformer_forward_flops_per_episode()
    elif model == "population":
        forward = population_forward_flops_per_episode(POPULATION_TRAIN_WORKERS)
    else:
        raise ValueError(f"unknown model: {model}")
    return 3 * forward * GLOBAL_BATCH_SIZE * OPTIMIZER_STEPS


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class PopulationRoutingProbe:
    """Aggregate exact top-k router statistics without changing model execution."""

    def __init__(self, model: PopulationLanguageOrganism, worker_count: int) -> None:
        if worker_count <= 0:
            raise ValueError("routing probe worker count must be positive")
        self.model = model
        self.worker_count = worker_count
        self.enabled = True
        self._pending_query: Tensor | None = None
        self._handles: list[Any] = []
        self.entropy_sum = 0.0
        self.decision_count = 0
        self.selection_counts = torch.zeros(worker_count, dtype=torch.int64)

    def __enter__(self) -> "PopulationRoutingProbe":
        self._handles = [
            self.model.router_query.register_forward_hook(self._query_hook),
            self.model.router_key.register_forward_hook(self._key_hook),
        ]
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._pending_query = None

    def _query_hook(self, module: nn.Module, inputs: tuple[Tensor, ...], output: Tensor) -> None:
        del module, inputs
        if self.enabled:
            self._pending_query = output.detach()

    def _key_hook(self, module: nn.Module, inputs: tuple[Tensor, ...], output: Tensor) -> None:
        del module, inputs
        if not self.enabled:
            self._pending_query = None
            return
        query = self._pending_query
        self._pending_query = None
        if query is None:
            raise RuntimeError("routing probe observed key projection without query projection")
        key = output.detach()
        if query.shape != key.shape or query.shape[1] != self.worker_count:
            raise RuntimeError("routing probe projection shape drifted")
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            self.model.config.router_dim
        )
        tie_break = -torch.arange(
            self.worker_count, device=scores.device, dtype=scores.dtype
        ) * 1e-7
        scores = scores + tie_break.view(1, 1, self.worker_count)
        selected_scores, selected_indices = torch.topk(
            scores,
            k=min(self.model.top_k, self.worker_count),
            dim=-1,
        )
        weights = torch.softmax(selected_scores.to(torch.float32), dim=-1)
        entropy = -(weights * torch.log(weights.clamp_min(1e-12))).sum(dim=-1)
        self.entropy_sum += float(entropy.sum().cpu().item())
        self.decision_count += entropy.numel()
        counts = torch.bincount(
            selected_indices.reshape(-1).to(torch.int64),
            minlength=self.worker_count,
        ).cpu()
        self.selection_counts += counts

    def summary(self) -> dict[str, float | int]:
        if self._pending_query is not None:
            raise RuntimeError("routing probe ended with an unmatched query projection")
        total = int(self.selection_counts.sum().item())
        if self.decision_count <= 0 or total <= 0:
            raise RuntimeError("routing probe captured no decisions")
        shares = self.selection_counts.to(torch.float64) / total
        positive = shares[shares > 0]
        sender_entropy = float((-(positive * torch.log(positive))).sum().item())
        effective_utilization = math.exp(sender_entropy) / self.worker_count
        mean = float(self.selection_counts.to(torch.float64).mean().item())
        standard_deviation = float(
            self.selection_counts.to(torch.float64).std(unbiased=False).item()
        )
        maximum_entropy = math.log(min(self.model.top_k, self.worker_count))
        mean_entropy = self.entropy_sum / self.decision_count
        return {
            "router_decisions": self.decision_count,
            "selected_messages": total,
            "mean_router_entropy_nats": mean_entropy,
            "normalized_router_entropy": (
                mean_entropy / maximum_entropy if maximum_entropy > 0 else 0.0
            ),
            "selected_sender_coverage": float(
                torch.count_nonzero(self.selection_counts).item() / self.worker_count
            ),
            "effective_worker_utilization": effective_utilization,
            "sender_selection_coefficient_of_variation": (
                standard_deviation / mean if mean > 0 else 0.0
            ),
        }


def _model_factory(name: str) -> nn.Module:
    if name == "transformer":
        return MatchedCausalTransformer()
    if name == "population":
        return PopulationLanguageOrganism(
            communication_rounds=POPULATION_COMMUNICATION_ROUNDS,
            top_k=POPULATION_TOP_K,
        )
    raise ValueError(f"unknown model: {name}")


def _set_optimizer_learning_rate(
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def _finite_float(value: Tensor | float, label: str) -> float:
    number = float(value.detach().to(torch.float64).cpu().item()) if isinstance(value, Tensor) else float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"reference training produced non-finite {label}")
    return number


def _train_model(
    *,
    name: str,
    seed: int,
    contract: TrainingContract,
    device: torch.device,
    output_root: pathlib.Path,
    schedule_sha256: str,
    dataset: ReferenceDatasetCache,
) -> tuple[nn.Module, dict[str, Any]]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = _model_factory(name).to(device)
    expected = (
        protocol.transformer_parameter_count()
        if name == "transformer"
        else protocol.population_parameter_count()
    )
    actual = count_parameters(model)
    if actual != expected:
        raise RuntimeError(f"{name} parameter count drifted: {actual} != {expected}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=PEAK_LEARNING_RATE,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
    )
    model.train()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    curves: list[dict[str, float | int]] = []
    target_tokens_per_episode: int | None = None
    progress_path = output_root / "progress" / f"{name}-seed-{seed}.json"
    progress_path.parent.mkdir(parents=True, exist_ok=True)

    for step in range(OPTIMIZER_STEPS):
        learning_rate = learning_rate_for_step(step)
        _set_optimizer_learning_rate(optimizer, learning_rate)
        optimizer.zero_grad(set_to_none=True)
        global_loss_sum = torch.zeros((), dtype=torch.float64, device=device)
        ordinals = training_ordinals(step)

        for start in range(0, GLOBAL_BATCH_SIZE, contract.microbatch):
            micro_ordinals = ordinals[start : start + contract.microbatch]
            batch = dataset.batch("train", micro_ordinals, device=device)
            if target_tokens_per_episode is None:
                first_target_tokens = int(batch.loss_mask.sum().item())
                if first_target_tokens % contract.microbatch:
                    raise RuntimeError("target-token count does not divide across episodes")
                target_tokens_per_episode = first_target_tokens // contract.microbatch
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(batch.input_ids)
                loss_sum = F.cross_entropy(
                    logits[batch.loss_mask].to(torch.float32),
                    batch.target_ids[batch.loss_mask],
                    reduction="sum",
                )
            scaled_loss = loss_sum / (
                GLOBAL_BATCH_SIZE * target_tokens_per_episode
            )
            scaled_loss.backward()
            global_loss_sum = global_loss_sum + loss_sum.detach().to(torch.float64)

        expected_tokens = GLOBAL_BATCH_SIZE * target_tokens_per_episode
        global_loss_value = _finite_float(global_loss_sum, "training loss")
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), GRADIENT_CLIP
        )
        gradient_norm_value = _finite_float(gradient_norm, "gradient norm")
        optimizer.step()

        update = step + 1
        if update == 1 or update % TRAINING_LOG_INTERVAL == 0 or update == OPTIMIZER_STEPS:
            torch.cuda.synchronize()
            curves.append(
                {
                    "optimizer_step": update,
                    "learning_rate": learning_rate,
                    "full_next_token_nll": global_loss_value / expected_tokens,
                    "gradient_norm_before_clip": gradient_norm_value,
                    "elapsed_seconds": time.perf_counter() - started,
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                }
            )
            progress_path.write_text(
                json.dumps(
                    {
                        "status": "IN_PROGRESS",
                        "version": VERSION,
                        "model": name,
                        "seed": seed,
                        "last_completed_optimizer_step": update,
                        "optimizer_steps": OPTIMIZER_STEPS,
                        "microbatch": contract.microbatch,
                        "gradient_accumulation_steps": contract.accumulation_steps,
                        "training_schedule_sha256": schedule_sha256,
                        "curves": curves,
                    },
                    indent=2,
                    sort_keys=True,
                ) + "\n",
                encoding="utf-8",
            )

    torch.cuda.synchronize()
    state_sha256 = canonical_state_sha256(model)
    checkpoint_path = output_root / "checkpoints" / f"{name}-seed-{seed}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    cpu_state = {
        key: value.detach().cpu().contiguous()
        for key, value in model.state_dict().items()
    }
    torch.save(
        {
            "version": VERSION,
            "model": name,
            "seed": seed,
            "optimizer_step": OPTIMIZER_STEPS,
            "state_dict": cpu_state,
        },
        checkpoint_path,
    )
    checkpoint_file_sha256 = _sha256(checkpoint_path)
    del cpu_state, optimizer

    row = {
        "model": name,
        "seed": seed,
        "parameter_count": actual,
        "optimizer_steps": OPTIMIZER_STEPS,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "microbatch": contract.microbatch,
        "gradient_accumulation_steps": contract.accumulation_steps,
        "training_schedule_sha256": schedule_sha256,
        "training_tokens": OPTIMIZER_STEPS * GLOBAL_BATCH_SIZE * target_tokens_per_episode,
        "estimated_training_flops": estimated_training_flops(name),
        "seconds": time.perf_counter() - started,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "canonical_checkpoint_sha256": state_sha256,
        "checkpoint_file": checkpoint_path.relative_to(output_root).as_posix(),
        "checkpoint_file_sha256": checkpoint_file_sha256,
        "curves": curves,
    }
    progress_path.write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "version": VERSION,
                "model": name,
                "seed": seed,
                "last_completed_optimizer_step": OPTIMIZER_STEPS,
                "canonical_checkpoint_sha256": state_sha256,
                "checkpoint_file_sha256": checkpoint_file_sha256,
                "curves": curves,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    return model, row


def _split_count(split: protocol.Split) -> int:
    if split == "validation":
        return protocol.REFERENCE_VALIDATION_EPISODES
    if split == "test":
        return protocol.REFERENCE_TEST_EPISODES
    raise ValueError("reference evaluation is restricted to validation/test")



def _answer_prompt_input_length(batch: LanguageBatch) -> int:
    if batch.answer_mask.ndim != 2 or batch.answer_mask.shape != batch.input_ids.shape:
        raise ValueError("answer mask/input shape mismatch")
    counts = batch.answer_mask.sum(dim=1)
    if not bool(torch.all(counts == ANSWER_TOKEN_COUNT)):
        raise ValueError("reference answer span must contain exactly five targets")
    first = torch.argmax(batch.answer_mask.to(torch.int64), dim=1)
    if not bool(torch.all(first == first[0])):
        raise ValueError("reference answer prompt length drifted within a batch")
    start = int(first[0].item())
    expected = torch.arange(
        start,
        start + ANSWER_TOKEN_COUNT,
        device=batch.answer_mask.device,
    )
    observed = torch.nonzero(batch.answer_mask[0], as_tuple=False).flatten()
    if not bool(torch.equal(observed, expected)):
        raise ValueError("reference answer mask is not one contiguous five-token span")
    prompt_length = start + 1
    if prompt_length + ANSWER_TOKEN_COUNT > protocol.MAX_SEQUENCE_LENGTH:
        raise ValueError("greedy answer generation exceeds the model sequence contract")
    return prompt_length


def greedy_answer_tokens(
    *,
    model: nn.Module,
    name: str,
    batch: LanguageBatch,
    input_ids: Tensor | None = None,
    worker_count: int | None = None,
) -> Tensor:
    """Generate the five answer tokens autoregressively without answer leakage."""
    if name == "population" and worker_count is None:
        raise ValueError("population greedy generation requires a worker count")
    if name == "transformer" and worker_count is not None:
        raise ValueError("transformer greedy generation cannot set a worker count")
    source = batch.input_ids if input_ids is None else input_ids
    if source.dtype != torch.long or source.shape != batch.input_ids.shape:
        raise ValueError("greedy generation input does not match the reference batch")
    prompt_length = _answer_prompt_input_length(batch)
    generated = source[:, :prompt_length].clone()
    answer: list[Tensor] = []
    for _ in range(ANSWER_TOKEN_COUNT):
        if name == "population":
            logits = model(generated, worker_count=worker_count)  # type: ignore[call-arg]
        elif name == "transformer":
            logits = model(generated)
        else:
            raise ValueError(f"unknown model: {name}")
        next_token = torch.argmax(logits[:, -1, :], dim=-1)
        answer.append(next_token)
        generated = torch.cat((generated, next_token.unsqueeze(1)), dim=1)
    return torch.stack(answer, dim=1)

def _evaluate_model(
    *,
    model: nn.Module,
    name: str,
    split: protocol.Split,
    evaluation_microbatch: int,
    checkpoint_sha256: str,
    dataset: ReferenceDatasetCache,
    worker_count: int | None = None,
) -> dict[str, Any]:
    count = _split_count(split)
    if name == "population" and worker_count is None:
        raise ValueError("population evaluation requires a worker count")
    if name == "transformer" and worker_count is not None:
        raise ValueError("transformer evaluation cannot set a worker count")

    model.eval()
    device = next(model.parameters()).device
    full_loss_sum = 0.0
    answer_loss_sum = 0.0
    target_tokens = 0
    answer_tokens = 0
    exact_correct = 0
    swapped_exact_correct = 0
    order_agreement_correct = 0
    color_correct = 0
    shape_correct = 0
    relation_correct = 0
    routed_messages = 0

    probe_context: contextlib.AbstractContextManager[Any]
    if name == "population":
        probe_context = PopulationRoutingProbe(
            model, int(worker_count)
        )  # type: ignore[arg-type]
    else:
        probe_context = contextlib.nullcontext(None)

    started = time.perf_counter()
    with torch.no_grad(), probe_context as probe:
        for start in range(0, count, evaluation_microbatch):
            ordinals = range(start, min(start + evaluation_microbatch, count))
            batch = dataset.batch(split, ordinals, device=device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                if name == "population":
                    forward = model.forward_with_state(  # type: ignore[attr-defined]
                        batch.input_ids,
                        worker_count=worker_count,
                    )
                    logits = forward.logits
                    routed_messages += int(forward.routed_messages)
                else:
                    logits = model(batch.input_ids)
            full_loss_sum += _finite_float(
                F.cross_entropy(
                    logits[batch.loss_mask].to(torch.float32),
                    batch.target_ids[batch.loss_mask],
                    reduction="sum",
                ),
                "evaluation full loss",
            )
            answer_loss_sum += _finite_float(
                F.cross_entropy(
                    logits[batch.answer_mask].to(torch.float32),
                    batch.target_ids[batch.answer_mask],
                    reduction="sum",
                ),
                "evaluation answer loss",
            )
            target_tokens += int(batch.loss_mask.sum().item())
            answer_tokens += int(batch.answer_mask.sum().item())

            answer_targets = batch.target_ids[batch.answer_mask].view(
                -1, ANSWER_TOKEN_COUNT
            )
            if probe is not None:
                probe.enabled = False
            try:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    answer_predictions = greedy_answer_tokens(
                        model=model,
                        name=name,
                        batch=batch,
                        worker_count=worker_count,
                    )
                    swapped_answer = greedy_answer_tokens(
                        model=model,
                        name=name,
                        batch=batch,
                        input_ids=swap_definition_order(batch.input_ids),
                        worker_count=worker_count,
                    )
            finally:
                if probe is not None:
                    probe.enabled = True

            exact_correct += int(
                torch.all(answer_predictions == answer_targets, dim=1).sum().item()
            )
            color_correct += int(
                (answer_predictions[:, (0, 3)] == answer_targets[:, (0, 3)]).sum().item()
            )
            shape_correct += int(
                (answer_predictions[:, (1, 4)] == answer_targets[:, (1, 4)]).sum().item()
            )
            relation_correct += int(
                (answer_predictions[:, 2] == answer_targets[:, 2]).sum().item()
            )
            swapped_exact_correct += int(
                torch.all(swapped_answer == answer_targets, dim=1).sum().item()
            )
            order_agreement_correct += int(
                (swapped_answer == answer_predictions).sum().item()
            )

    torch.cuda.synchronize()
    nll = full_loss_sum / target_tokens
    answer_nll = answer_loss_sum / answer_tokens
    teacher_forced_flops = (
        transformer_forward_flops_per_episode()
        if name == "transformer"
        else population_forward_flops_per_episode(int(worker_count))
    )
    prompt_length = _answer_prompt_input_length(
        dataset.batch(split, (0,), device=device)
    )
    greedy_answer_flops = sum(
        (
            transformer_forward_flops_per_episode(sequence_length=prompt_length + offset)
            if name == "transformer"
            else population_forward_flops_per_episode(
                int(worker_count), sequence_length=prompt_length + offset
            )
        )
        for offset in range(ANSWER_TOKEN_COUNT)
    )
    result: dict[str, Any] = {
        "split": split,
        "episodes": count,
        "checkpoint_sha256": checkpoint_sha256,
        "next_token_nll": nll,
        "perplexity": math.exp(nll),
        "answer_span_nll": answer_nll,
        "answer_exact_accuracy": exact_correct / count,
        "color_token_accuracy": color_correct / (2 * count),
        "shape_token_accuracy": shape_correct / (2 * count),
        "relation_token_accuracy": relation_correct / count,
        "swapped_definition_answer_exact_accuracy": swapped_exact_correct / count,
        "definition_order_answer_token_agreement": order_agreement_correct / (ANSWER_TOKEN_COUNT * count),
        "estimated_forward_flops_per_episode": teacher_forced_flops,
        "estimated_greedy_answer_flops_per_episode": greedy_answer_flops,
        "answer_exact_accuracy_per_gigaflop": (
            (exact_correct / count) / (greedy_answer_flops / 1e9)
        ),
        "answer_exact_decoding": "AUTOREGRESSIVE_GREEDY_FIVE_TOKEN",
        "seconds": time.perf_counter() - started,
    }
    if name == "population":
        routing = probe.summary()  # type: ignore[union-attr]
        processed_tokens = count * (protocol.MAX_SEQUENCE_LENGTH - 1)
        result.update(
            {
                "worker_count": worker_count,
                "routed_messages": routed_messages,
                "routed_messages_per_processed_token": routed_messages / processed_tokens,
                "routed_messages_per_episode": routed_messages / count,
                "persistent_state_bytes_per_episode_bf16": (
                    int(worker_count) * protocol.PopulationConfig().worker_width * 2
                ),
                "routing": routing,
            }
        )
    return result


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _valid_evaluation(
    row: object,
    *,
    split: str,
    episodes: int,
    checkpoint_sha256: object,
    worker_count: int | None = None,
) -> bool:
    if not isinstance(row, dict):
        return False
    if (
        row.get("split") != split
        or row.get("episodes") != episodes
        or row.get("checkpoint_sha256") != checkpoint_sha256
    ):
        return False
    nonnegative = (
        "next_token_nll",
        "perplexity",
        "answer_span_nll",
        "estimated_forward_flops_per_episode",
        "estimated_greedy_answer_flops_per_episode",
        "answer_exact_accuracy_per_gigaflop",
        "seconds",
    )
    if any(not _finite_number(row.get(field)) or float(row[field]) < 0 for field in nonnegative):
        return False
    if row.get("answer_exact_decoding") != "AUTOREGRESSIVE_GREEDY_FIVE_TOKEN":
        return False
    accuracies = (
        "answer_exact_accuracy",
        "color_token_accuracy",
        "shape_token_accuracy",
        "relation_token_accuracy",
        "swapped_definition_answer_exact_accuracy",
        "definition_order_answer_token_agreement",
    )
    if any(
        not _finite_number(row.get(field)) or not 0.0 <= float(row[field]) <= 1.0
        for field in accuracies
    ):
        return False
    if worker_count is None:
        return "worker_count" not in row
    if row.get("worker_count") != worker_count:
        return False
    population_fields = (
        "routed_messages",
        "routed_messages_per_processed_token",
        "routed_messages_per_episode",
        "persistent_state_bytes_per_episode_bf16",
    )
    if any(not _finite_number(row.get(field)) or float(row[field]) <= 0 for field in population_fields):
        return False
    routing = row.get("routing")
    if not isinstance(routing, dict):
        return False
    routing_nonnegative = (
        "router_decisions",
        "selected_messages",
        "mean_router_entropy_nats",
        "normalized_router_entropy",
        "selected_sender_coverage",
        "effective_worker_utilization",
        "sender_selection_coefficient_of_variation",
    )
    if any(
        not _finite_number(routing.get(field)) or float(routing[field]) < 0
        for field in routing_nonnegative
    ):
        return False
    for field in (
        "normalized_router_entropy",
        "selected_sender_coverage",
        "effective_worker_utilization",
    ):
        if float(routing[field]) > 1.0 + 1e-12:
            return False
    return True


def classify(seed_rows: list[dict[str, Any]]) -> str:
    if [row.get("seed") for row in seed_rows] != list(protocol.INITIALIZATION_SEEDS):
        raise ValueError("reference result seeds are incomplete or out of order")
    valid = True
    expected_schedule = training_schedule_sha256()
    observed_microbatch: int | None = None
    expected_training_tokens = (
        OPTIMIZER_STEPS * GLOBAL_BATCH_SIZE * EXPECTED_TARGET_TOKENS_PER_EPISODE
    )
    for seed_row in seed_rows:
        transformer = seed_row.get("transformer", {})
        population = seed_row.get("population", {})
        if transformer.get("parameter_count") != protocol.transformer_parameter_count():
            valid = False
        if population.get("parameter_count") != protocol.population_parameter_count():
            valid = False
        for trained in (transformer, population):
            microbatch = trained.get("microbatch")
            microbatch_valid = (
                type(microbatch) is int
                and microbatch > 0
                and GLOBAL_BATCH_SIZE % microbatch == 0
            )
            if not microbatch_valid:
                valid = False
                expected_accumulation = None
            else:
                expected_accumulation = GLOBAL_BATCH_SIZE // microbatch
                if observed_microbatch is None:
                    observed_microbatch = microbatch
                elif microbatch != observed_microbatch:
                    valid = False
            if (
                trained.get("optimizer_steps") != OPTIMIZER_STEPS
                or trained.get("global_batch_size") != GLOBAL_BATCH_SIZE
                or trained.get("gradient_accumulation_steps")
                != expected_accumulation
                or trained.get("training_tokens") != expected_training_tokens
                or trained.get("training_schedule_sha256") != expected_schedule
            ):
                valid = False
            if not _is_sha256(trained.get("canonical_checkpoint_sha256")):
                valid = False
            if not _is_sha256(trained.get("checkpoint_file_sha256")):
                valid = False
            for field in (
                "estimated_training_flops",
                "seconds",
                "peak_allocated_bytes",
                "peak_reserved_bytes",
            ):
                if not _finite_number(trained.get(field)) or float(trained[field]) <= 0:
                    valid = False

        transformer_hash = transformer.get("canonical_checkpoint_sha256")
        validation = transformer.get("validation", {})
        if not _valid_evaluation(
            validation,
            split="validation",
            episodes=protocol.REFERENCE_VALIDATION_EPISODES,
            checkpoint_sha256=transformer_hash,
        ):
            valid = False
        elif validation.get("answer_exact_accuracy", -1.0) < 0.95:
            valid = False
        if not _valid_evaluation(
            transformer.get("test", {}),
            split="test",
            episodes=protocol.REFERENCE_TEST_EPISODES,
            checkpoint_sha256=transformer_hash,
        ):
            valid = False

        population_hash = population.get("canonical_checkpoint_sha256")
        for split_name, split, episodes in (
            ("validation_by_workers", "validation", protocol.REFERENCE_VALIDATION_EPISODES),
            ("test_by_workers", "test", protocol.REFERENCE_TEST_EPISODES),
        ):
            rows = population.get(split_name, {})
            if list(rows) != [str(worker) for worker in protocol.EVAL_WORKERS]:
                valid = False
                continue
            for worker in protocol.EVAL_WORKERS:
                if not _valid_evaluation(
                    rows[str(worker)],
                    split=split,
                    episodes=episodes,
                    checkpoint_sha256=population_hash,
                    worker_count=worker,
                ):
                    valid = False
    return PASS if valid else INVALID


def population_scaling_summary(seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if classify(seed_rows) != PASS:
        return {"conclusion": SCALING_INVALID}
    aggregates: dict[str, dict[str, float]] = {}
    for worker in protocol.EVAL_WORKERS:
        test_rows = [
            seed_row["population"]["test_by_workers"][str(worker)]
            for seed_row in seed_rows
        ]
        aggregates[str(worker)] = {
            "mean_answer_exact_accuracy": sum(
                row["answer_exact_accuracy"] for row in test_rows
            ) / len(test_rows),
            "mean_answer_span_nll": sum(
                row["answer_span_nll"] for row in test_rows
            ) / len(test_rows),
            "mean_forward_flops_per_episode": sum(
                row["estimated_forward_flops_per_episode"] for row in test_rows
            ) / len(test_rows),
            "mean_effective_worker_utilization": sum(
                row["routing"]["effective_worker_utilization"] for row in test_rows
            ) / len(test_rows),
        }
    first = aggregates[str(protocol.EVAL_WORKERS[0])]
    last = aggregates[str(protocol.EVAL_WORKERS[-1])]
    accuracy_gain = (
        last["mean_answer_exact_accuracy"] - first["mean_answer_exact_accuracy"]
    )
    nll_reduction = (
        (first["mean_answer_span_nll"] - last["mean_answer_span_nll"])
        / first["mean_answer_span_nll"]
        if first["mean_answer_span_nll"] > 0
        else 0.0
    )
    nondegrading: list[dict[str, Any]] = []
    for lower, upper in zip(protocol.EVAL_WORKERS, protocol.EVAL_WORKERS[1:]):
        lower_accuracy = aggregates[str(lower)]["mean_answer_exact_accuracy"]
        upper_accuracy = aggregates[str(upper)]["mean_answer_exact_accuracy"]
        nondegrading.append(
            {
                "transition": f"{lower}->{upper}",
                "accuracy_delta": upper_accuracy - lower_accuracy,
                "nondegrading_within_0_5_points": upper_accuracy >= lower_accuracy - 0.005,
            }
        )
    nondegrading_count = sum(
        row["nondegrading_within_0_5_points"] for row in nondegrading
    )
    magnitude_pass = accuracy_gain >= 0.05 or nll_reduction >= 0.10
    monotonicity_pass = nondegrading_count >= 3
    transformer_test = sum(
        seed_row["transformer"]["test"]["answer_exact_accuracy"]
        for seed_row in seed_rows
    ) / len(seed_rows)
    population_32_test = aggregates["32"]["mean_answer_exact_accuracy"]
    return {
        "conclusion": (
            SCALING_SUPPORTS
            if magnitude_pass and monotonicity_pass
            else SCALING_DOES_NOT_SUPPORT
        ),
        "aggregates_by_worker": aggregates,
        "accuracy_gain_256_minus_16": accuracy_gain,
        "answer_nll_reduction_fraction_256_vs_16": nll_reduction,
        "magnitude_criterion_passed": magnitude_pass,
        "consecutive_transitions": nondegrading,
        "nondegrading_transition_count": nondegrading_count,
        "monotonicity_criterion_passed": monotonicity_pass,
        "mean_transformer_test_answer_exact_accuracy": transformer_test,
        "mean_population_32_test_answer_exact_accuracy": population_32_test,
        "population_32_minus_transformer_accuracy": population_32_test - transformer_test,
    }


def run(
    output_root: pathlib.Path,
    execution_head: str,
    *,
    microbatch: int = DEFAULT_MICROBATCH,
    evaluation_microbatch: int = DEFAULT_EVALUATION_MICROBATCH,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"reference training output exists: {output_root}")
    if len(execution_head) != 40 or any(
        character not in "0123456789abcdef" for character in execution_head
    ):
        raise ValueError("reference training execution head is malformed")
    contract = validate_contract(microbatch, evaluation_microbatch)
    if not torch.cuda.is_available():
        raise RuntimeError("reference training requires CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("reference training requires CUDA BF16 support")

    torch.cuda.set_device(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda", 0)
    schedule_sha256 = training_schedule_sha256()
    output_root.mkdir(parents=True)
    start_path = output_root / "run-start.json"
    start_payload = {
        "status": STATUS,
        "phase": "BUILDING_DATASET_CACHE",
        "version": VERSION,
        "branch": BRANCH,
        "base_head": BASE_HEAD,
        "execution_head": execution_head,
        "training_schedule_sha256": schedule_sha256,
        "contract": contract.__dict__,
        "seeds": list(protocol.INITIALIZATION_SEEDS),
        "dataset_fingerprints_first_256": {
            split: protocol.dataset_fingerprint(split, 256)
            for split in ("train", "validation", "test")
        },
    }
    start_path.write_text(
        json.dumps(start_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    dataset = ReferenceDatasetCache.build(device)
    start_payload.update(
        {
            "phase": "TRAINING",
            "dataset_cache_build_seconds": dataset.build_seconds,
            "dataset_cache_resident_bytes": dataset.resident_bytes,
        }
    )
    start_path.write_text(
        json.dumps(start_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    seed_rows: list[dict[str, Any]] = []
    for seed in protocol.INITIALIZATION_SEEDS:
        transformer, transformer_train = _train_model(
            name="transformer",
            seed=seed,
            contract=contract,
            device=device,
            output_root=output_root,
            schedule_sha256=schedule_sha256,
            dataset=dataset,
        )
        transformer_hash = transformer_train["canonical_checkpoint_sha256"]
        transformer_train["validation"] = _evaluate_model(
            model=transformer,
            name="transformer",
            split="validation",
            evaluation_microbatch=contract.evaluation_microbatch,
            checkpoint_sha256=transformer_hash,
            dataset=dataset,
        )
        transformer_train["test"] = _evaluate_model(
            model=transformer,
            name="transformer",
            split="test",
            evaluation_microbatch=contract.evaluation_microbatch,
            checkpoint_sha256=transformer_hash,
            dataset=dataset,
        )
        del transformer
        torch.cuda.empty_cache()

        population, population_train = _train_model(
            name="population",
            seed=seed,
            contract=contract,
            device=device,
            output_root=output_root,
            schedule_sha256=schedule_sha256,
            dataset=dataset,
        )
        population_hash = population_train["canonical_checkpoint_sha256"]
        validation_by_workers: dict[str, dict[str, Any]] = {}
        test_by_workers: dict[str, dict[str, Any]] = {}
        for worker_count in protocol.EVAL_WORKERS:
            validation_by_workers[str(worker_count)] = _evaluate_model(
                model=population,
                name="population",
                split="validation",
                evaluation_microbatch=contract.evaluation_microbatch,
                checkpoint_sha256=population_hash,
                dataset=dataset,
                worker_count=worker_count,
            )
            test_by_workers[str(worker_count)] = _evaluate_model(
                model=population,
                name="population",
                split="test",
                evaluation_microbatch=contract.evaluation_microbatch,
                checkpoint_sha256=population_hash,
                dataset=dataset,
                worker_count=worker_count,
            )
        population_train["validation_by_workers"] = validation_by_workers
        population_train["test_by_workers"] = test_by_workers
        del population
        torch.cuda.empty_cache()

        seed_row = {
            "seed": seed,
            "transformer": transformer_train,
            "population": population_train,
        }
        seed_rows.append(seed_row)
        (output_root / f"seed-{seed}.json").write_text(
            json.dumps(seed_row, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    diagnosis = classify(seed_rows)
    summary = {
        "status": STATUS,
        "version": VERSION,
        "branch": BRANCH,
        "base_head": BASE_HEAD,
        "execution_head": execution_head,
        "diagnosis": diagnosis,
        "training_schedule_sha256": schedule_sha256,
        "contract": contract.__dict__,
        "dataset_cache_build_seconds": dataset.build_seconds,
        "dataset_cache_resident_bytes": dataset.resident_bytes,
        "dataset_fingerprints_first_256": {
            split: protocol.dataset_fingerprint(split, 256)
            for split in ("train", "validation", "test")
        },
        "cuda": {
            "device_name": torch.cuda.get_device_name(0),
            "device_capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "bf16_supported": torch.cuda.is_bf16_supported(),
        },
        "seed_rows": seed_rows,
        "population_scaling": population_scaling_summary(seed_rows),
        "boundaries": {
            "full_next_token_objective_only": True,
            "answer_span_training_weighted": False,
            "fixed_final_checkpoint_used": True,
            "test_used_for_checkpoint_selection": False,
            "population_trained_only_at_32_workers": True,
            "same_population_checkpoint_used_at_all_worker_counts": True,
            "worker_specific_learned_parameters_used": False,
            "gate9_evidence_modified": False,
        },
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
