"""Run the frozen three-seed relay-v1 confirmation protocol.

Scientific negative results are valid outputs. This program exits nonzero only for execution,
contract, or provenance failures; a failed confirmation Gate is written to the result artifact
and still returns success so CI cannot hide a negative scientific result.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from ai_hypothesis.population_compute.collective_relay import COLLECTIVE_RELAY_VERSION
from ai_hypothesis.population_compute.confirmation_gate_v1 import (
    FROZEN_CONFIRMATION_BATCH_SIZE,
    FROZEN_CONFIRMATION_WORLD_COUNT,
    assess_confirmation_gate_v1,
)
from ai_hypothesis.population_compute.relay_experiment_v1 import (
    RELAY_EXPERIMENT_V1,
    RelayDevelopmentResultV1,
    RelayTrainingConfigV1,
    assess_relay_results_v1,
    evaluate_relay_split_v1,
    load_relay_checkpoint_v1,
    save_relay_checkpoint_v1,
    train_relay_model_v1,
)
from ai_hypothesis.population_compute.relay_protocol_v1 import RELAY_PROTOCOL_VERSION


CONFIRMATION_TRAINING_SEEDS: tuple[int, ...] = (1, 2, 3)
CONFIRMATION_CONFIG = RelayTrainingConfigV1()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute the frozen relay-v1 confirmation protocol for seeds 1/2/3."
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-dir",
        default="results/population_compute_scaling_v1/confirmation",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    provenance = _provenance(args.device)
    (output_root / "execution-provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    results: list[RelayDevelopmentResultV1] = []
    seed_summaries: list[dict[str, object]] = []

    for training_seed in CONFIRMATION_TRAINING_SEEDS:
        seed_dir = output_root / f"seed_{training_seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        model, training = train_relay_model_v1(
            training_seed=training_seed,
            config=CONFIRMATION_CONFIG,
            device=args.device,
        )
        checkpoint = save_relay_checkpoint_v1(
            model,
            training,
            CONFIRMATION_CONFIG,
            seed_dir / "model-v1.pt",
        )

        loaded_model, checkpoint_payload = load_relay_checkpoint_v1(
            checkpoint,
            device=args.device,
        )
        _validate_checkpoint_boundary(
            loaded_model=loaded_model,
            checkpoint_payload=checkpoint_payload,
            training=training,
        )

        evaluations = evaluate_relay_split_v1(
            loaded_model,
            training_seed=training_seed,
            split="confirmation",
            world_count=FROZEN_CONFIRMATION_WORLD_COUNT,
            batch_size=FROZEN_CONFIRMATION_BATCH_SIZE,
            device=args.device,
            allow_confirmation=True,
        )
        result = RelayDevelopmentResultV1(
            experiment_version=RELAY_EXPERIMENT_V1,
            protocol_version=RELAY_PROTOCOL_VERSION,
            benchmark_version=COLLECTIVE_RELAY_VERSION,
            evaluation_split="confirmation",
            confirmation_opened=True,
            training=training,
            training_config=CONFIRMATION_CONFIG,
            evaluation_world_count=FROZEN_CONFIRMATION_WORLD_COUNT,
            evaluation_batch_size=FROZEN_CONFIRMATION_BATCH_SIZE,
            evaluations=evaluations,
            assessments=assess_relay_results_v1(evaluations),
        )
        seed_payload = result.to_dict()
        seed_payload["provenance"] = {
            **provenance,
            "checkpoint": str(checkpoint),
            "evaluated_parameter_fingerprint": loaded_model.parameter_fingerprint(),
            "training_seed": training_seed,
        }
        result_path = seed_dir / "confirmation.json"
        result_path.write_text(
            json.dumps(seed_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        results.append(result)
        seed_summaries.append(
            {
                "training_seed": training_seed,
                "checkpoint": str(checkpoint),
                "parameter_fingerprint": training.parameter_fingerprint,
                "learned_parameter_count": training.learned_parameter_count,
                "initial_total_loss": training.initial_total_loss,
                "final_total_loss": training.final_total_loss,
                "mean_last_50_total_loss": training.mean_last_50_total_loss,
                "result": str(result_path),
            }
        )

    gate = assess_confirmation_gate_v1(tuple(results))
    gate_payload = gate.to_dict()
    gate_payload["protocol"] = {
        "experiment_version": RELAY_EXPERIMENT_V1,
        "protocol_version": RELAY_PROTOCOL_VERSION,
        "benchmark_version": COLLECTIVE_RELAY_VERSION,
        "training_seeds": list(CONFIRMATION_TRAINING_SEEDS),
        "training_config": {
            "steps": CONFIRMATION_CONFIG.steps,
            "batch_size": CONFIRMATION_CONFIG.batch_size,
            "learning_rate": CONFIRMATION_CONFIG.learning_rate,
            "weight_decay": CONFIRMATION_CONFIG.weight_decay,
            "gradient_clip_norm": CONFIRMATION_CONFIG.gradient_clip_norm,
            "gate_supervision_weight": CONFIRMATION_CONFIG.gate_supervision_weight,
            "state_width": CONFIRMATION_CONFIG.model.state_width,
            "message_width": CONFIRMATION_CONFIG.model.message_width,
        },
        "evaluation_world_count": FROZEN_CONFIRMATION_WORLD_COUNT,
        "evaluation_batch_size": FROZEN_CONFIRMATION_BATCH_SIZE,
        "execution_mode": "eager",
    }
    gate_payload["provenance"] = provenance
    gate_payload["seed_summaries"] = seed_summaries
    gate_path = output_root / "confirmation-gate.json"
    gate_path.write_text(
        json.dumps(gate_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "confirmation_gate": str(gate_path),
                "passes_gate": gate.passes_gate,
                "training_seeds": list(CONFIRMATION_TRAINING_SEEDS),
                "seed_summaries": seed_summaries,
                "scientific_negative_is_valid_execution": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _validate_checkpoint_boundary(*, loaded_model, checkpoint_payload, training) -> None:
    if loaded_model.parameter_fingerprint() != training.parameter_fingerprint:
        raise RuntimeError("reloaded confirmation checkpoint changed parameter fingerprint")
    if loaded_model.trainable_parameter_count() != training.learned_parameter_count:
        raise RuntimeError("reloaded confirmation checkpoint changed parameter count")
    if checkpoint_payload.get("experiment_version") != RELAY_EXPERIMENT_V1:
        raise RuntimeError("confirmation checkpoint lost experiment-version provenance")
    if checkpoint_payload.get("protocol_version") != RELAY_PROTOCOL_VERSION:
        raise RuntimeError("confirmation checkpoint lost protocol-version provenance")
    if checkpoint_payload.get("benchmark_version") != COLLECTIVE_RELAY_VERSION:
        raise RuntimeError("confirmation checkpoint lost benchmark-version provenance")
    saved_summary = checkpoint_payload.get("training_summary")
    if not isinstance(saved_summary, dict):
        raise RuntimeError("confirmation checkpoint lost training summary")
    if saved_summary.get("parameter_fingerprint") != training.parameter_fingerprint:
        raise RuntimeError("saved confirmation summary fingerprint differs from checkpoint")


def _provenance(device: str) -> dict[str, object]:
    return {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "requested_device": device,
        "execution_mode": "eager",
        "confirmation_opened": True,
    }


if __name__ == "__main__":
    raise SystemExit(main())
