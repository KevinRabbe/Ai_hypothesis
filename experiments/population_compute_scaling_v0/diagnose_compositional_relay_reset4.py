"""Development-only fixed-four relay-2 diagnostic with worker-state reset per hop.

The shared field remains recurrent. Local worker state is reinitialized from the same local
record before each hop by invoking the shared cell for one round at a time. This isolates
persistent local-state interference without adding learned parameters or changing messages.
"""

from __future__ import annotations

import argparse
import json
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
    NODE_BIT_WIDTH,
    RelayPopulationConfig,
    RelayPopulationModel,
    build_relay_tensor_batch,
    decode_node_logits,
)


ACTIVE_WORKERS = 4
TRAINING_WORLD_SEED_BASE = 9_000_000_000
HELDOUT_WORLD_SEED_BASE = 10_000_000_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=505)
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
        raise RuntimeError("reset diagnostic no longer matches frozen relay-2 scope")

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
        )
        batch = build_relay_tensor_batch(
            worlds,
            active_workers=ACTIVE_WORKERS,
            device=device,
        )
        output = _forward_with_state_reset(
            model,
            batch,
            communication_mode=CommunicationMode.SPARSE_SHARED_V0,
        )
        targets = batch.target_bits.gt(0).to(dtype=output.logits.dtype)
        loss = F.binary_cross_entropy_with_logits(output.logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("reset diagnostic produced non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))

    model.eval()
    heldout_worlds = _threshold_worlds(
        base_seed=HELDOUT_WORLD_SEED_BASE,
        offset=0,
        count=args.heldout_world_count,
    )
    heldout: dict[str, object] = {}
    for mode in (
        CommunicationMode.NO_COMMUNICATION,
        CommunicationMode.SPARSE_SHARED_V0,
    ):
        exact = 0
        correct_bits = 0
        total_bits = 0
        with torch.inference_mode():
            for offset in range(0, len(heldout_worlds), 32):
                worlds = heldout_worlds[offset : offset + 32]
                batch = build_relay_tensor_batch(
                    worlds,
                    active_workers=ACTIVE_WORKERS,
                    device=device,
                )
                if not bool(torch.all(batch.information_complete).item()):
                    raise RuntimeError("reset heldout batch must be information-complete")
                output = _forward_with_state_reset(
                    model,
                    batch,
                    communication_mode=mode,
                )
                predictions = decode_node_logits(output.logits)
                exact += int((predictions == batch.answer_keys).sum().item())
                predicted_bits = output.logits >= 0
                target_bits = batch.target_bits > 0
                correct_bits += int((predicted_bits == target_bits).sum().item())
                total_bits += int(target_bits.numel())
        heldout[mode.value] = {
            "exact_solve_rate": exact / len(heldout_worlds),
            "bit_accuracy": correct_bits / total_bits,
        }

    tail = losses[-min(100, len(losses)) :]
    payload = {
        "diagnostic": "compositional-relay-2-reset4-v0",
        "architecture": "shared-node-projection-message-content",
        "local_state_policy": "reset_each_hop",
        "seed": args.seed,
        "device": str(device),
        "active_workers": ACTIVE_WORKERS,
        "scope_threshold": ACTIVE_WORKERS,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "parameter_count": model.trainable_parameter_count(),
        "parameter_fingerprint": model.parameter_fingerprint(),
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


def _forward_with_state_reset(
    model: RelayPopulationModel,
    batch,
    *,
    communication_mode: CommunicationMode,
):
    seed = torch.tanh(model.query_projection(batch.start_bits))
    value_bits = batch.local_inputs[..., NODE_BIT_WIDTH:]
    message_content = torch.tanh(model.query_projection(value_bits))
    first = model.cell(
        batch.local_inputs,
        batch.active_mask,
        recurrent_rounds=1,
        communication_mode=communication_mode,
        shared_seed=seed,
        message_content=message_content,
    )
    return model.cell(
        batch.local_inputs,
        batch.active_mask,
        recurrent_rounds=1,
        communication_mode=communication_mode,
        shared_seed=first.final_shared,
        message_content=message_content,
    )


def _threshold_worlds(*, base_seed: int, offset: int, count: int) -> tuple:
    difficulty = RELAY_DIFFICULTIES[0]
    thresholds = relay_scope_thresholds(difficulty)
    threshold = ACTIVE_WORKERS
    selected = []
    seed = base_seed + offset * len(thresholds)
    while len(selected) < count:
        if thresholds[seed % len(thresholds)] == threshold:
            world = generate_relay_world(seed, difficulty)
            if world.scope_threshold != threshold:
                raise RuntimeError("reset diagnostic threshold selection drifted")
            selected.append(world)
        seed += 1
    return tuple(selected)


if __name__ == "__main__":
    raise SystemExit(main())
