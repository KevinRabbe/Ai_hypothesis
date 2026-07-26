"""Development-only relay-2 diagnostic for parameter-free competitive aggregation.

Each width trains a fresh checkpoint with the #64 relay protocol plus per-sample softmax
normalization of active worker gate logits. This tests whether scale-dependent distractor
accumulation is the fixed-width bottleneck. It is not Gate-v0 and opens no confirmation data.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from ai_hypothesis.population_compute.collective_relay import (
    RELAY_DIFFICULTIES,
    generate_relay_world,
    relay_scope_thresholds,
)
from ai_hypothesis.population_compute.contract import CommunicationMode
from ai_hypothesis.population_compute.relay_model import (
    RelayPopulationConfig,
    RelayPopulationModel,
    build_relay_tensor_batch,
    decode_node_logits,
)


WIDTHS = (4, 16, 64, 256)
TRAIN_BATCH_BY_WIDTH = {4: 64, 16: 16, 64: 4, 256: 1}
TRAINING_SEED_BASE = 15_000_000_000
HELDOUT_SEED_BASE = 16_000_000_000


@dataclass(frozen=True)
class WidthResult:
    width: int
    training_seed: int
    train_batch_size: int
    steps: int
    learned_parameter_count: int
    parameter_fingerprint: str
    first_loss: float
    best_loss: float
    final_loss: float
    mean_last_100_loss: float
    sparse_exact_solve_rate: float
    sparse_bit_accuracy: float
    no_communication_exact_solve_rate: float
    no_communication_bit_accuracy: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--heldout-world-count", type=int, default=512)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.steps <= 0 or args.heldout_world_count <= 0:
        raise SystemExit("steps and heldout world count must be positive")

    difficulty = RELAY_DIFFICULTIES[0]
    if difficulty.name != "relay-2":
        raise RuntimeError("competitive diagnostic expects relay-2")
    thresholds = relay_scope_thresholds(difficulty)
    if any(width not in thresholds for width in WIDTHS):
        raise RuntimeError("competitive widths must remain frozen relay-2 scope thresholds")

    device = torch.device(args.device)
    results = tuple(
        _train_width(
            width=width,
            training_seed=800 + index,
            steps=args.steps,
            heldout_world_count=args.heldout_world_count,
            device=device,
        )
        for index, width in enumerate(WIDTHS)
    )
    if len({result.learned_parameter_count for result in results}) != 1:
        raise RuntimeError("competitive diagnostic changed learned parameter count across widths")

    payload = {
        "diagnostic": "relay-competitive-aggregation-v0",
        "architecture": "compositional-node-messages+hop-local-state+softmax-worker-competition",
        "aggregation": "softmax(message_gate_logits) over active workers per sample",
        "difficulty": difficulty.name,
        "device": str(device),
        "steps_per_width": args.steps,
        "heldout_world_count_per_width": args.heldout_world_count,
        "active_worker_updates_per_training_batch": {
            str(width): width * TRAIN_BATCH_BY_WIDTH[width] for width in WIDTHS
        },
        "results": [asdict(result) for result in results],
        "confirmation_opened": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _train_width(
    *,
    width: int,
    training_seed: int,
    steps: int,
    heldout_world_count: int,
    device: torch.device,
) -> WidthResult:
    batch_size = TRAIN_BATCH_BY_WIDTH[width]
    torch.manual_seed(training_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(training_seed)

    model = RelayPopulationModel(RelayPopulationConfig(state_width=64, message_width=24)).to(device)
    parameter_count = model.trainable_parameter_count()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    losses: list[float] = []

    model.train()
    train_base = TRAINING_SEED_BASE + width * 10_000_000
    for step in range(steps):
        worlds = _threshold_worlds(
            base_seed=train_base,
            offset=step * batch_size,
            count=batch_size,
            threshold=width,
        )
        batch = build_relay_tensor_batch(worlds, active_workers=width, device=device)
        output = model(
            batch,
            communication_mode=CommunicationMode.SPARSE_SHARED_V0,
            recurrent_rounds=2,
        )
        targets = batch.target_bits.gt(0).to(dtype=output.logits.dtype)
        loss = F.binary_cross_entropy_with_logits(output.logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"width {width} produced non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))

    fingerprint = model.parameter_fingerprint()
    heldout = _threshold_worlds(
        base_seed=HELDOUT_SEED_BASE + width * 10_000_000,
        offset=0,
        count=heldout_world_count,
        threshold=width,
    )
    sparse_exact, sparse_bits = _evaluate(
        model,
        heldout,
        width=width,
        communication_mode=CommunicationMode.SPARSE_SHARED_V0,
        device=device,
    )
    no_comm_exact, no_comm_bits = _evaluate(
        model,
        heldout,
        width=width,
        communication_mode=CommunicationMode.NO_COMMUNICATION,
        device=device,
    )
    if model.parameter_fingerprint() != fingerprint:
        raise RuntimeError(f"width {width} evaluation mutated the checkpoint")

    tail = losses[-min(100, len(losses)) :]
    return WidthResult(
        width=width,
        training_seed=training_seed,
        train_batch_size=batch_size,
        steps=steps,
        learned_parameter_count=parameter_count,
        parameter_fingerprint=fingerprint,
        first_loss=losses[0],
        best_loss=min(losses),
        final_loss=losses[-1],
        mean_last_100_loss=sum(tail) / len(tail),
        sparse_exact_solve_rate=sparse_exact,
        sparse_bit_accuracy=sparse_bits,
        no_communication_exact_solve_rate=no_comm_exact,
        no_communication_bit_accuracy=no_comm_bits,
    )


def _evaluate(
    model: RelayPopulationModel,
    worlds: tuple,
    *,
    width: int,
    communication_mode: CommunicationMode,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    exact = 0
    correct_bits = 0
    total_bits = 0
    batch_size = max(1, 256 // width)
    with torch.inference_mode():
        for offset in range(0, len(worlds), batch_size):
            world_batch = worlds[offset : offset + batch_size]
            batch = build_relay_tensor_batch(world_batch, active_workers=width, device=device)
            output = model(
                batch,
                communication_mode=communication_mode,
                recurrent_rounds=2,
            )
            predictions = decode_node_logits(output.logits)
            exact += int((predictions == batch.answer_keys).sum().item())
            predicted_bits = output.logits >= 0
            target_bits = batch.target_bits > 0
            correct_bits += int((predicted_bits == target_bits).sum().item())
            total_bits += int(target_bits.numel())
    return exact / len(worlds), correct_bits / total_bits


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
                raise RuntimeError("competitive threshold selection drifted")
            selected.append(world)
        seed += 1
    return tuple(selected)


if __name__ == "__main__":
    raise SystemExit(main())
