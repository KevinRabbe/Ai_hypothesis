"""Development-only relay-2 diagnostic for the compositional message protocol.

This repeats the failed relay-2 slice from closed PR #63 on the same deterministic
seed domains. It is not Gate-v0 evaluation and must not open confirmation data.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from ai_hypothesis.population_compute.collective_relay import (
    RELAY_DIFFICULTIES,
    RelayDifficulty,
    generate_relay_dataset,
    generate_relay_world,
    relay_scope_thresholds,
)
from ai_hypothesis.population_compute.contract import CommunicationMode
from ai_hypothesis.population_compute.relay_experiment import evaluate_relay_condition
from ai_hypothesis.population_compute.relay_model import (
    RelayPopulationConfig,
    RelayPopulationModel,
    build_relay_tensor_batch,
)


TRAINING_POPULATIONS = (4, 16, 64, 256)
TRAINING_WORLD_SEED_BASE = 5_000_000_000
HELDOUT_WORLD_SEED_BASE = 6_000_000_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=303)
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--heldout-world-count", type=int, default=512)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.steps <= 0 or args.batch_size <= 0 or args.heldout_world_count <= 0:
        raise SystemExit("steps, batch size and heldout world count must be positive")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    difficulty = RELAY_DIFFICULTIES[0]
    if difficulty.name != "relay-2":
        raise RuntimeError("expected relay-2 to remain the first frozen difficulty")

    model = RelayPopulationModel(
        RelayPopulationConfig(state_width=64, message_width=24)
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    losses: list[float] = []

    model.train()
    for step in range(args.steps):
        active_workers = TRAINING_POPULATIONS[step % len(TRAINING_POPULATIONS)]
        population_round = step // len(TRAINING_POPULATIONS)
        thresholds = tuple(
            threshold
            for threshold in relay_scope_thresholds(difficulty)
            if threshold <= active_workers
        )
        threshold = thresholds[population_round % len(thresholds)]
        worlds = _relay_worlds_for_threshold(
            step=step,
            batch_size=args.batch_size,
            difficulty=difficulty,
            threshold=threshold,
        )
        batch = build_relay_tensor_batch(
            worlds,
            active_workers=active_workers,
            device=device,
        )
        if not bool(torch.all(batch.information_complete).item()):
            raise RuntimeError("relay-2 diagnostic training batch is not information-complete")

        output = model(
            batch,
            communication_mode=CommunicationMode.SPARSE_SHARED_V0,
            recurrent_rounds=2,
        )
        targets = batch.target_bits.gt(0).to(dtype=output.logits.dtype)
        loss = F.binary_cross_entropy_with_logits(output.logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("relay-2 diagnostic produced non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))

    model.eval()
    worlds = generate_relay_dataset(
        start_seed=HELDOUT_WORLD_SEED_BASE,
        world_count=args.heldout_world_count,
        difficulty=difficulty,
    )
    fingerprint = model.parameter_fingerprint()
    parameter_count = model.trainable_parameter_count()
    heldout: dict[str, object] = {}
    for population in TRAINING_POPULATIONS:
        by_mode: dict[str, object] = {}
        for mode in (
            CommunicationMode.NO_COMMUNICATION,
            CommunicationMode.SPARSE_SHARED_V0,
        ):
            evaluation = evaluate_relay_condition(
                model,
                worlds,
                training_seed=args.seed,
                benchmark_seed=6_000,
                active_workers=population,
                communication_mode=mode,
                batch_size=32,
                device=device,
            )
            metrics = evaluation.metrics
            if metrics.learned_parameter_count != parameter_count:
                raise RuntimeError("diagnostic changed learned parameter count")
            if metrics.parameter_fingerprint != fingerprint:
                raise RuntimeError("diagnostic changed parameter fingerprint")
            by_mode[mode.value] = {
                "solve_rate": metrics.solve_rate,
                "information_complete_rate": metrics.information_complete_rate,
                "solve_rate_given_information_complete": (
                    metrics.solve_rate_given_information_complete
                ),
                "solve_rate_given_information_incomplete": (
                    metrics.solve_rate_given_information_incomplete
                ),
            }
        heldout[str(population)] = by_mode

    tail = losses[-min(100, len(losses)) :]
    payload = {
        "diagnostic": "compositional-relay-2-v0",
        "architecture": "shared-node-projection-message-content",
        "seed": args.seed,
        "device": str(device),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "parameter_count": parameter_count,
        "parameter_fingerprint": fingerprint,
        "first_loss": losses[0],
        "best_loss": min(losses),
        "final_loss": losses[-1],
        "mean_last_100_loss": sum(tail) / len(tail),
        "heldout": heldout,
    }
    if any(
        not math.isfinite(float(payload[key]))
        for key in ("first_loss", "best_loss", "final_loss", "mean_last_100_loss")
    ):
        raise RuntimeError("diagnostic loss summary is non-finite")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _relay_worlds_for_threshold(
    *,
    step: int,
    batch_size: int,
    difficulty: RelayDifficulty,
    threshold: int,
) -> tuple:
    thresholds = relay_scope_thresholds(difficulty)
    segment_size = batch_size * len(thresholds)
    start = TRAINING_WORLD_SEED_BASE + step * segment_size
    selected = tuple(
        seed
        for seed in range(start, start + segment_size)
        if thresholds[seed % len(thresholds)] == threshold
    )[:batch_size]
    if len(selected) != batch_size:
        raise RuntimeError("could not fill threshold-matched diagnostic batch")
    worlds = tuple(generate_relay_world(seed, difficulty) for seed in selected)
    if any(world.scope_threshold != threshold for world in worlds):
        raise RuntimeError("diagnostic threshold selection drifted")
    return worlds


if __name__ == "__main__":
    raise SystemExit(main())
