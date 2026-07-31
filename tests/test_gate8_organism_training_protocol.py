from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest


PROTOCOL_PATH = pathlib.Path(
    "ai_hypothesis/population_compute/gate8_organism_training_protocol.py"
)
WORLD_PATH = pathlib.Path(
    "ai_hypothesis/population_compute/gate8_distributed_transformation_worlds.py"
)


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load Gate8 module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


p = _load(PROTOCOL_PATH, "gate8_training_protocol_test_module")
w = _load(WORLD_PATH, "gate8_training_protocol_test_world_contract")


def _candidate(
    *,
    step: int,
    target: float,
    message: float = 1.0,
    activity: float = 1.0,
    loss: float = 0.1,
    inbox_coverage: int = 256,
    target_coverage: int = 256,
):
    return p.Gate8CheckpointCandidate(
        step=step,
        conditions=tuple(
            p.Gate8ValidationConditionRow(
                population=population,
                depth=depth,
                target_accuracy=target,
            )
            for population, depth in p.GATE8_TRAINING_CONDITIONS
        ),
        message_accuracy=message,
        activity_accuracy=activity,
        validation_loss=loss,
        inbox_code_coverage=inbox_coverage,
        target_code_coverage=target_coverage,
    )


class Gate8OrganismTrainingProtocolTests(unittest.TestCase):
    def test_message_code_round_trip_covers_exactly_256_codes(self) -> None:
        observed = set()
        for carrier in range(16):
            for symbol in range(16):
                code = p.gate8_encode_message_code(
                    carrier=carrier,
                    symbol=symbol,
                )
                self.assertEqual(
                    p.gate8_decode_message_code(code),
                    (carrier, symbol),
                )
                observed.add(code)
        self.assertEqual(observed, set(range(256)))

    def test_every_primitive_transform_induces_a_256_code_bijection(self) -> None:
        for transform in w.GATE8_TRANSFORM_PERMUTATIONS:
            outputs = set()
            for inbox_code in range(256):
                carrier, input_symbol = p.gate8_decode_message_code(inbox_code)
                output_symbol = transform[input_symbol]
                output_code = p.gate8_target_message_code(
                    inbox_code=inbox_code,
                    output_symbol=output_symbol,
                    source_is_root=False,
                    root_symbol=0,
                )
                output_carrier, decoded_symbol = p.gate8_decode_message_code(output_code)
                self.assertEqual(output_carrier, (carrier + 1) % 16)
                self.assertEqual(decoded_symbol, output_symbol)
                outputs.add(output_code)
            self.assertEqual(outputs, set(range(256)))

    def test_root_message_uses_public_root_symbol_as_initial_carrier(self) -> None:
        for root_symbol in range(16):
            output_symbol = (root_symbol + 5) % 16
            code = p.gate8_target_message_code(
                inbox_code=0,
                output_symbol=output_symbol,
                source_is_root=True,
                root_symbol=root_symbol,
            )
            self.assertEqual(
                p.gate8_decode_message_code(code),
                ((root_symbol + 1) % 16, output_symbol),
            )
        with self.assertRaisesRegex(ValueError, "seed code"):
            p.gate8_target_message_code(
                inbox_code=1,
                output_symbol=0,
                source_is_root=True,
                root_symbol=0,
            )

    def test_training_schedule_is_exact_uniform_and_nonadaptive(self) -> None:
        self.assertEqual(p.GATE8_TRAINING_WORLDS_PER_SEED, 262_144)
        self.assertEqual(p.GATE8_TRAINING_WORLD_BATCH_SIZE, 256)
        self.assertEqual(p.GATE8_OPTIMIZER_STEPS, 1_024)
        observed_first_cycle = tuple(
            (
                p.gate8_training_world_address(index).population,
                p.gate8_training_world_address(index).depth,
                p.gate8_training_world_address(index).condition_world_index,
            )
            for index in range(6)
        )
        self.assertEqual(
            observed_first_cycle,
            tuple(
                (population, depth, 0)
                for population, depth in p.GATE8_TRAINING_CONDITIONS
            ),
        )
        counts = p.gate8_condition_world_counts()
        self.assertEqual(
            tuple(counts.values()),
            (43_691, 43_691, 43_691, 43_691, 43_690, 43_690),
        )
        self.assertEqual(sum(counts.values()), 262_144)

    def test_learning_rate_schedule_has_exact_warmup_and_cosine_end(self) -> None:
        self.assertAlmostEqual(
            p.gate8_learning_rate(1),
            p.GATE8_LEARNING_RATE / p.GATE8_WARMUP_STEPS,
        )
        self.assertEqual(
            p.gate8_learning_rate(p.GATE8_WARMUP_STEPS),
            p.GATE8_LEARNING_RATE,
        )
        self.assertAlmostEqual(
            p.gate8_learning_rate(p.GATE8_OPTIMIZER_STEPS),
            p.GATE8_MINIMUM_LEARNING_RATE,
        )
        self.assertGreater(
            p.gate8_learning_rate(512),
            p.gate8_learning_rate(768),
        )
        with self.assertRaisesRegex(ValueError, "optimizer step"):
            p.gate8_learning_rate(0)

    def test_checkpoint_selection_uses_frozen_lexicographic_order(self) -> None:
        candidates = (
            _candidate(step=256, target=0.98, message=1.0, loss=0.01),
            _candidate(step=512, target=0.99, message=0.998, loss=0.2),
            _candidate(step=768, target=0.99, message=0.999, loss=0.3),
            _candidate(step=1024, target=0.99, message=0.999, loss=0.1),
        )
        selected = p.select_gate8_checkpoint(candidates)
        self.assertEqual(selected.step, 1_024)

        exact_tie = tuple(
            _candidate(step=step, target=1.0, message=1.0, activity=1.0, loss=0.0)
            for step in p.GATE8_CHECKPOINT_STEPS
        )
        self.assertEqual(p.select_gate8_checkpoint(exact_tie).step, 256)

    def test_training_admission_requires_every_frozen_guard(self) -> None:
        admitted = tuple(
            _candidate(step=step, target=1.0)
            for step in p.GATE8_CHECKPOINT_STEPS
        )
        self.assertEqual(
            p.classify_gate8_training(admitted),
            p.GATE8_TRAINING_ADMITTED,
        )

        low_condition = list(
            _candidate(step=1_024, target=1.0).conditions
        )
        low_condition[-1] = p.Gate8ValidationConditionRow(
            population=128,
            depth=16,
            target_accuracy=0.989,
        )
        failed = (
            _candidate(step=256, target=0.9),
            _candidate(step=512, target=0.95),
            _candidate(step=768, target=0.98),
            p.Gate8CheckpointCandidate(
                step=1_024,
                conditions=tuple(low_condition),
                message_accuracy=1.0,
                activity_accuracy=1.0,
                validation_loss=0.0,
                inbox_code_coverage=256,
                target_code_coverage=256,
            ),
        )
        self.assertEqual(
            p.classify_gate8_training(failed),
            p.GATE8_TRAINING_NOT_ADMITTED,
        )

        missing_coverage = tuple(
            _candidate(
                step=step,
                target=1.0,
                inbox_coverage=255,
            )
            for step in p.GATE8_CHECKPOINT_STEPS
        )
        self.assertEqual(
            p.classify_gate8_training(missing_coverage),
            p.GATE8_TRAINING_NOT_ADMITTED,
        )

    def test_protocol_plan_preserves_held_out_composition_depths(self) -> None:
        plan = p.gate8_organism_training_protocol_plan()
        self.assertEqual(
            plan["runtime_head"],
            "1a2be148411bc71ba35fda12b035b724f06ec166",
        )
        self.assertEqual(plan["training_seeds"], [0, 1, 2])
        self.assertEqual(plan["training_worlds_per_seed"], 262_144)
        self.assertEqual(plan["world_batch_size"], 256)
        self.assertEqual(plan["optimizer_steps"], 1_024)
        self.assertEqual(plan["checkpoint_steps"], [256, 512, 768, 1_024])
        self.assertEqual(plan["validation_namespace"], "validation")
        self.assertEqual(plan["validation_world_indices"], [0, 511])
        self.assertTrue(
            all(depth <= 16 for _population, depth in p.GATE8_TRAINING_CONDITIONS)
        )
        self.assertFalse(plan["execution_admitted"])
        self.assertFalse(plan["training_admitted"])
        self.assertFalse(plan["checkpoint_write_admitted"])
        self.assertFalse(plan["scientific_test_worlds_admitted"])
        self.assertFalse(plan["reference_model_admitted"])

    def test_protocol_module_has_no_execution_or_external_model_surface(self) -> None:
        text = PROTOCOL_PATH.read_text(encoding="utf-8")
        for token in (
            "import torch",
            "generate_gate8_world",
            "torch.optim",
            "optimizer.step",
            "backward(",
            "torch.save",
            "AutoModel",
            "AutoTokenizer",
            "from_pretrained",
            "snapshot_download",
            "model.safetensors",
            "split=\"test\"",
        ):
            self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main()
