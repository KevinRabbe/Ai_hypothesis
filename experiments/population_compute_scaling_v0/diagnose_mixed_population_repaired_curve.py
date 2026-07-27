"""Development-only one-checkpoint population curve with the two localized relay repairs.

The learned architecture remains corrected #64. Training adds the fixed oracle-available gate loss
from #69, generalized across every relay hop. Communication uses parameter-free softmax-normalized
competition during training and communicating evaluation. One persisted/reloaded checkpoint is
reused at 1/4/16/64/256 active states and against the no-communication control.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from ai_hypothesis.population_compute.collective_relay import (
    RELAY_DIFFICULTIES,
    RelayDifficulty,
    RelayWorld,
    generate_relay_dataset,
    relay_scope_thresholds,
)
from ai_hypothesis.population_compute.contract import (
    DEVELOPMENT_POPULATION_SIZES,
    CommunicationMode,
)
from ai_hypothesis.population_compute.relay_experiment import (
    DEVELOPMENT_SEED_START,
    RelayTrainingConfig,
    training_world_batch,
)
from ai_hypothesis.population_compute.relay_model import (
    NODE_BIT_WIDTH,
    RelayPopulationConfig,
    RelayPopulationModel,
    build_relay_tensor_batch,
    decode_node_logits,
    encode_node_bits,
)


GATE_LOSS_WEIGHT = 1.0


@dataclass(frozen=True)
class TrainingSummary:
    training_seed: int
    steps: int
    batch_size: int
    examples_seen: int
    initial_total_loss: float
    final_total_loss: float
    mean_last_50_total_loss: float
    final_relay_loss: float
    final_gate_loss: float
    learned_parameter_count: int
    parameter_fingerprint: str


@dataclass(frozen=True)
class ConditionResult:
    difficulty: str
    active_workers: int
    communication: str
    task_count: int
    solved_count: int
    solve_rate: float
    bit_accuracy: float
    information_complete_count: int
    information_complete_rate: float
    solved_information_complete_count: int
    solve_rate_given_information_complete: float | None
    solved_information_incomplete_count: int
    solve_rate_given_information_incomplete: float | None
    scope_cohorts: tuple[dict[str, object], ...]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--evaluation-world-count", type=int, default=1000)
    parser.add_argument("--evaluation-batch-size", type=int, default=64)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = RelayTrainingConfig(
        steps=args.steps,
        batch_size=args.train_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        model=RelayPopulationConfig(state_width=64, message_width=24),
    )
    config.validate()
    if args.evaluation_world_count <= 0 or args.evaluation_batch_size <= 0:
        raise SystemExit("evaluation counts must be positive")

    device = torch.device(args.device)
    model, training = _train(
        training_seed=args.training_seed,
        config=config,
        device=device,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "model-state.pt"
    torch.save(model.state_dict(), checkpoint)

    loaded = RelayPopulationModel(config.model).to(device)
    loaded.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    if loaded.trainable_parameter_count() != training.learned_parameter_count:
        raise RuntimeError("persisted checkpoint changed learned parameter count")
    if loaded.parameter_fingerprint() != training.parameter_fingerprint:
        raise RuntimeError("persisted checkpoint changed parameter fingerprint")

    conditions = _evaluate_curve(
        loaded,
        training_seed=args.training_seed,
        world_count=args.evaluation_world_count,
        batch_size=args.evaluation_batch_size,
        device=device,
    )
    if loaded.parameter_fingerprint() != training.parameter_fingerprint:
        raise RuntimeError("population-curve evaluation mutated the checkpoint")

    payload = {
        "diagnostic": "mixed-population-repaired-curve-v0",
        "evaluation_split": "development",
        "confirmation_opened": False,
        "population_sizes": list(DEVELOPMENT_POPULATION_SIZES),
        "training_repairs": {
            "gate_supervision_weight": GATE_LOSS_WEIGHT,
            "gate_supervision": "unique correct worker at every oracle-known training hop",
            "communicating_reducer": "softmax-normalized gate competition",
            "oracle_used_at_inference": False,
        },
        "training_config": {
            "steps": config.steps,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "gradient_clip_norm": config.gradient_clip_norm,
            "model": asdict(config.model),
        },
        "training": asdict(training),
        "checkpoint": str(checkpoint),
        "conditions": [asdict(condition) for condition in conditions],
        "curve_summary": _curve_summary(conditions),
        "interpretation_frozen_before_result": {
            "increasing_raw_and_strong_complete_conditional": "one fixed checkpoint converts increased runtime population into capability once information becomes available",
            "raw_increases_but_complete_conditional_collapses": "scope grows but learned population computation is not population-stable",
            "communication_no_better_than_no_comm": "shared communication does not add usable capability",
        },
    }
    result_path = output_dir / "development.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "checkpoint": str(checkpoint),
        "result": str(result_path),
        "learned_parameter_count": training.learned_parameter_count,
        "parameter_fingerprint": training.parameter_fingerprint,
        "confirmation_opened": False,
    }, indent=2, sort_keys=True))
    return 0


def _train(
    *,
    training_seed: int,
    config: RelayTrainingConfig,
    device: torch.device,
) -> tuple[RelayPopulationModel, TrainingSummary]:
    torch.manual_seed(training_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(training_seed)
    model = RelayPopulationModel(config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    losses: list[float] = []
    final_relay_loss = 0.0
    final_gate_loss = 0.0
    model.train()

    for step in range(config.steps):
        difficulty = RELAY_DIFFICULTIES[step % len(RELAY_DIFFICULTIES)]
        thresholds = relay_scope_thresholds(difficulty)
        threshold_cycle = step // len(RELAY_DIFFICULTIES)
        active_workers = thresholds[threshold_cycle % len(thresholds)]
        worlds = training_world_batch(
            training_seed=training_seed,
            step=step,
            difficulty=difficulty,
            active_workers=active_workers,
            batch_size=config.batch_size,
        )
        batch = build_relay_tensor_batch(
            worlds,
            active_workers=active_workers,
            device=device,
        )
        if not bool(torch.all(batch.information_complete).item()):
            raise RuntimeError("mixed training batch contains incomplete worlds")

        logits = _normalized_forward(model, batch, rounds=difficulty.hop_count)
        targets = batch.target_bits.gt(0).to(dtype=logits.dtype)
        relay_loss = F.binary_cross_entropy_with_logits(logits, targets)
        gate_loss = _oracle_gate_loss(
            model,
            batch,
            worlds,
            active_workers=active_workers,
            device=device,
        )
        total_loss = relay_loss + GATE_LOSS_WEIGHT * gate_loss
        if not torch.isfinite(total_loss):
            raise RuntimeError("mixed repaired training produced non-finite loss")

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        optimizer.step()

        losses.append(float(total_loss.detach().item()))
        final_relay_loss = float(relay_loss.detach().item())
        final_gate_loss = float(gate_loss.detach().item())

    fingerprint = model.parameter_fingerprint()
    return model, TrainingSummary(
        training_seed=training_seed,
        steps=config.steps,
        batch_size=config.batch_size,
        examples_seen=config.steps * config.batch_size,
        initial_total_loss=losses[0],
        final_total_loss=losses[-1],
        mean_last_50_total_loss=sum(losses[-50:]) / len(losses[-50:]),
        final_relay_loss=final_relay_loss,
        final_gate_loss=final_gate_loss,
        learned_parameter_count=model.trainable_parameter_count(),
        parameter_fingerprint=fingerprint,
    )


def _normalized_forward(model, batch, *, rounds: int) -> torch.Tensor:
    active_workers = batch.active_workers
    local = batch.local_inputs[:, :active_workers, :]
    batch_size = local.shape[0]
    flat_local = local.reshape(batch_size * active_workers, -1)
    initial = torch.tanh(model.cell.input_projection(flat_local))
    content = model.query_projection(local[..., NODE_BIT_WIDTH:])
    shared = torch.tanh(model.query_projection(batch.start_bits))
    states = initial

    for _ in range(rounds):
        shared_flat = (
            shared.unsqueeze(1)
            .expand(batch_size, active_workers, model.config.message_width)
            .reshape(batch_size * active_workers, model.config.message_width)
        )
        states = model.cell.update(torch.cat((flat_local, shared_flat), dim=-1), initial)
        gate_logits = model.cell.message_gate(states).reshape(batch_size, active_workers)
        weights = torch.softmax(gate_logits, dim=1)
        shared = torch.tanh((content * weights.unsqueeze(-1)).sum(dim=1))

    pooled = states.reshape(batch_size, active_workers, model.config.state_width).mean(dim=1)
    return model.cell.output_head(model.cell.output_norm(torch.cat((pooled, shared), dim=-1)))


def _oracle_gate_loss(
    model,
    batch,
    worlds: tuple[RelayWorld, ...],
    *,
    active_workers: int,
    device: torch.device,
) -> torch.Tensor:
    local = batch.local_inputs[:, :active_workers, :]
    batch_size = local.shape[0]
    flat_local = local.reshape(batch_size * active_workers, -1)
    initial = torch.tanh(model.cell.input_projection(flat_local))
    queries, target_slots = _oracle_hop_targets(worlds, device=device)
    losses: list[torch.Tensor] = []

    for hop in range(queries.shape[1]):
        clean_query = torch.tanh(model.query_projection(encode_node_bits(queries[:, hop])))
        shared_flat = (
            clean_query.unsqueeze(1)
            .expand(batch_size, active_workers, model.config.message_width)
            .reshape(batch_size * active_workers, model.config.message_width)
        )
        states = model.cell.update(torch.cat((flat_local, shared_flat), dim=-1), initial)
        gate_logits = model.cell.message_gate(states).reshape(batch_size, active_workers)
        losses.append(F.cross_entropy(gate_logits, target_slots[:, hop]))
    return torch.stack(losses).mean()


def _oracle_hop_targets(
    worlds: tuple[RelayWorld, ...],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    hop_count = worlds[0].difficulty.hop_count
    query_rows: list[list[int]] = []
    slot_rows: list[list[int]] = []
    for world in worlds:
        if world.difficulty.hop_count != hop_count:
            raise ValueError("oracle gate batch mixed relay depths")
        query = world.start_key
        queries: list[int] = []
        slots: list[int] = []
        for _ in range(hop_count):
            matches = tuple(
                record
                for record in world.records
                if record.is_chain_edge and record.key == query
            )
            if len(matches) != 1:
                raise RuntimeError("expected one oracle chain record per relay hop")
            record = matches[0]
            queries.append(query)
            slots.append(record.worker_slot)
            query = record.value
        if query != world.answer_key:
            raise RuntimeError("oracle chain targets do not terminate at answer")
        query_rows.append(queries)
        slot_rows.append(slots)
    return (
        torch.tensor(query_rows, dtype=torch.int64, device=device),
        torch.tensor(slot_rows, dtype=torch.int64, device=device),
    )


def _evaluate_curve(
    model,
    *,
    training_seed: int,
    world_count: int,
    batch_size: int,
    device: torch.device,
) -> tuple[ConditionResult, ...]:
    del training_seed
    model.eval()
    results: list[ConditionResult] = []
    for difficulty in RELAY_DIFFICULTIES:
        worlds = generate_relay_dataset(
            start_seed=DEVELOPMENT_SEED_START,
            world_count=world_count,
            difficulty=difficulty,
        )
        for communication in ("normalized_shared", "no_communication"):
            for active_workers in DEVELOPMENT_POPULATION_SIZES:
                results.append(
                    _evaluate_condition(
                        model,
                        worlds,
                        difficulty=difficulty,
                        active_workers=active_workers,
                        communication=communication,
                        batch_size=batch_size,
                        device=device,
                    )
                )
    return tuple(results)


def _evaluate_condition(
    model,
    worlds: tuple[RelayWorld, ...],
    *,
    difficulty: RelayDifficulty,
    active_workers: int,
    communication: str,
    batch_size: int,
    device: torch.device,
) -> ConditionResult:
    solved_count = 0
    correct_bits = 0
    total_bits = 0
    complete_count = 0
    solved_complete = 0
    solved_incomplete = 0
    cohort_counts = {
        threshold: [0, 0] for threshold in relay_scope_thresholds(difficulty)
    }

    with torch.inference_mode():
        for offset in range(0, len(worlds), batch_size):
            rows = worlds[offset : offset + batch_size]
            batch = build_relay_tensor_batch(
                rows,
                active_workers=active_workers,
                device=device,
            )
            if communication == "normalized_shared":
                logits = _normalized_forward(model, batch, rounds=difficulty.hop_count)
            elif communication == "no_communication":
                logits = model(
                    batch,
                    communication_mode=CommunicationMode.NO_COMMUNICATION,
                    recurrent_rounds=difficulty.hop_count,
                ).logits
            else:
                raise ValueError("unsupported communication mode")

            predicted = decode_node_logits(logits)
            solved = predicted.eq(batch.answer_keys)
            complete = batch.information_complete
            solved_count += int(solved.sum().item())
            complete_count += int(complete.sum().item())
            solved_complete += int((solved & complete).sum().item())
            solved_incomplete += int((solved & ~complete).sum().item())
            target_bits = batch.target_bits > 0
            predicted_bits = logits >= 0
            correct_bits += int(predicted_bits.eq(target_bits).sum().item())
            total_bits += int(target_bits.numel())

            solved_rows = tuple(bool(value) for value in solved.cpu().tolist())
            for world, row_solved in zip(rows, solved_rows, strict=True):
                cohort_counts[world.scope_threshold][0] += 1
                cohort_counts[world.scope_threshold][1] += int(row_solved)

    task_count = len(worlds)
    incomplete_count = task_count - complete_count
    cohorts = tuple(
        {
            "scope_threshold": threshold,
            "task_count": cohort_counts[threshold][0],
            "solved_count": cohort_counts[threshold][1],
            "solve_rate": (
                cohort_counts[threshold][1] / cohort_counts[threshold][0]
                if cohort_counts[threshold][0]
                else 0.0
            ),
        }
        for threshold in sorted(cohort_counts)
    )
    return ConditionResult(
        difficulty=difficulty.name,
        active_workers=active_workers,
        communication=communication,
        task_count=task_count,
        solved_count=solved_count,
        solve_rate=solved_count / task_count,
        bit_accuracy=correct_bits / total_bits,
        information_complete_count=complete_count,
        information_complete_rate=complete_count / task_count,
        solved_information_complete_count=solved_complete,
        solve_rate_given_information_complete=(
            solved_complete / complete_count if complete_count else None
        ),
        solved_information_incomplete_count=solved_incomplete,
        solve_rate_given_information_incomplete=(
            solved_incomplete / incomplete_count if incomplete_count else None
        ),
        scope_cohorts=cohorts,
    )


def _curve_summary(conditions: tuple[ConditionResult, ...]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for difficulty in (difficulty.name for difficulty in RELAY_DIFFICULTIES):
        normalized = sorted(
            (
                row
                for row in conditions
                if row.difficulty == difficulty and row.communication == "normalized_shared"
            ),
            key=lambda row: row.active_workers,
        )
        no_comm = sorted(
            (
                row
                for row in conditions
                if row.difficulty == difficulty and row.communication == "no_communication"
            ),
            key=lambda row: row.active_workers,
        )
        summary[difficulty] = {
            "population_sizes": [row.active_workers for row in normalized],
            "normalized_solve_rates": [row.solve_rate for row in normalized],
            "normalized_information_complete_rates": [
                row.information_complete_rate for row in normalized
            ],
            "normalized_solve_given_complete": [
                row.solve_rate_given_information_complete for row in normalized
            ],
            "no_communication_solve_rates": [row.solve_rate for row in no_comm],
            "endpoint_gain": normalized[-1].solve_rate - normalized[0].solve_rate,
            "communication_endpoint_advantage": (
                normalized[-1].solve_rate - no_comm[-1].solve_rate
            ),
            "nondecreasing_raw_steps": sum(
                later.solve_rate >= earlier.solve_rate
                for earlier, later in zip(normalized, normalized[1:])
            ),
        }
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
