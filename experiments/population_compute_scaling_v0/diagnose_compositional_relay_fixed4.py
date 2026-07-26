"""Development-only relay-2 diagnostic at one fixed four-worker population.

This removes mixed-population training interference from the compositional relay variant.
It is not Gate-v0 evaluation and uses only development seed domains.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from ai_hypothesis.population_compute.collective_relay import (
    RELAY_DIFFICULTIES,
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


ACTIVE_WORKERS = 4
TRAINING_WORLD_SEED_BASE = 7_000_000_000
HELDOUT_WORLD_SEED_BASE = 8_000_000_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=404)
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--heldout-world-count", type=int, default=512)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.steps <= 0 or args.batch_size <= 0 or args.heldout_world_count <= 0:
        raise SystemExit("steps, batch size and heldout world count must be positive")

    difficulty = RELAY_DIFFICULTIES[0]
    thresholds = relay_scope_thresholds(difficulty)
    if difficulty.name != "relay-2" or thresholds[0] != ACTIVE_WORKERS:
        raise RuntimeError("fixed-4 diagnostic no longer matches frozen relay-2 scope")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model = RelayPopulationModel(
        RelayPopulationConfig(state_width=64, message_width=24)
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    losses: list[float] = []

    model.train()
    for step in range(args.steps):
        worlds = _threshold_worlds(
            base_seed=TRAINING_WORLD_SEED_BASE,
            offset=step * args.batch_size,
            count=args.batch_size,
            threshold=ACTIVE_WORKERS,
        )
        batch = build_relay_tensor_batch(
            worlds,
            active_workers=ACTIVE_WORKERS,
            device=device,
        )
        if not bool(torch.all(batch.information_complete).item()):
            raise RuntimeError("fixed-4 training batch is not information-complete")
        output = model(
            batch,
            communication_mode=CommunicationMode.SPARSE_SHARED_V0,
            recurrent_rounds=2,
        )
        targets = batch.target_bits.gt(0).to(dtype=output.logits.dtype)
        loss = F.binary_cross_entropy_with_logits(output.logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("fixed-4 diagnostic produced non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))

    model.eval()
    heldout_worlds = generate_relay_dataset(
        start_seed=HELDOUT_WORLD_SEED_BASE,
        world_count=args.heldout_world_count,
        difficulty=difficulty,
    )
    fingerprint = model.parameter_fingerprint()
    parameter_count = model.trainable_parameter_count()
    heldout: dict[str, object] = {}
    for mode in (
        CommunicationMode.NO_COMMUNICATION,
        CommunicationMode.SPARSE_SHARED_V0,
    ):
        evaluation = evaluate_relay_condition(
            model,
            heldout_worlds,
            training_seed=args.seed,
            benchmark_seed=8_000,
            active_workers=ACTIVE_WORKERS,
            communication_mode=mode,
            batch_size=32,
            device=device,
        )
        metrics = evaluation.metrics
        heldout[mode.value] = {
            "solve_rate": metrics.solve_rate,
            "information_complete_rate": metrics.information_complete_rate,
            "solve_rate_given_information_complete": (
                metrics.solve_rate_given_information_complete
            ),
        }

    tail = losses[-min(100, len(losses)) :]
    payload = {
        "diagnostic": "compositional-relay-2-fixed4-v0",
        "architecture": "shared-node-projection-message-content",
        "seed": args.seed,
        "device": str(device),
        "active_workers": ACTIVE_WORKERS,
        "scope_threshold": ACTIVE_WORKERS,
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
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _threshold_worlds(
    *,
    base_seed: int,
    offset: int,
    count: int,
    threshold: int,
) -> tuple:
    difficulty = RELAY_DIFFICULTIES[0]
    thresholds = relay_scope_thresholds(difficulty)
    selected = []
    seed = base_seed + offset * len(thresholds)
    while len(selected) < count:
        if thresholds[seed % len(thresholds)] == threshold:
            world = generate_relay_world(seed, difficulty)
            if world.scope_threshold != threshold:
                raise RuntimeError("fixed diagnostic threshold selection drifted")
            selected.append(world)
        seed += 1
    return tuple(selected)


if __name__ == "__main__":
    raise SystemExit(main())
