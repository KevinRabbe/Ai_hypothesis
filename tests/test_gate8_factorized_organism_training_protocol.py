from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest


PROTOCOL_PATH = pathlib.Path(
    "ai_hypothesis/population_compute/"
    "gate8_factorized_organism_training_protocol.py"
)
WORLD_PATH = pathlib.Path(
    "ai_hypothesis/population_compute/"
    "gate8_distributed_transformation_worlds.py"
)


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load Gate8 v1 module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


p = _load(PROTOCOL_PATH, "gate8_v1_training_protocol_test_module")
w = _load(WORLD_PATH, "gate8_v1_training_protocol_world_contract")


def _candidate(
    *,
    step: int,
    target: float,
    exact_message: float = 1.0,
    carrier: float = 1.0,
    symbol: float = 1.0,
    loss: float = 0.1,
    inbox_coverage: int = 256,
    target_coverage: int = 256,
    target_carrier_coverage: int = 16,
    target_symbol_coverage: int = 16,
):
    return p.Gate8V1CheckpointCandidate(
        step=step,
        conditions=tuple(
            p.Gate8V1ValidationConditionRow(
                population=population,
                depth=depth,
                target_accuracy=target,
            )
            for population, depth in p.GATE8_V1_TRAINING_CONDITIONS
        ),
        exact_message_accuracy=exact_message,
        carrier_accuracy=carrier,
        symbol_accuracy=symbol,
        validation_loss=loss,
        inbox_code_coverage=inbox_coverage,
        target_code_coverage=target_coverage,
        target_carrier_coverage=target_carrier_coverage,
        target_symbol_coverage=target_symbol_coverage,
    )


class Gate8V1FactorizedTrainingProtocolTests(unittest.TestCase):
    def test_message_code_round_trip_covers_exactly_256_codes(self) -> None:
        observed = set()
        for carrier in range(16):
            for symbol in range(16):
                code = p.gate8_v1_encode_message_code(
                    carrier=carrier,
                    symbol=symbol,
                )
                self.assertEqual(
                    p.gate8_v1_decode_message_code(code),
                    (carrier, symbol),
                )
                observed.add(code)
        self.assertEqual(observed, set(range(256)))

    def test_every_transform_induces_exact_factorized_256_code_bijection(self) -> None:
        for transform in w.GATE8_TRANSFORM_PERMUTATIONS:
            outputs = set()
            for inbox_code in range(256):
                input_carrier, input_symbol = p.gate8_v1_decode_message_code(
                    inbox_code
                )
                carrier, symbol, code = p.gate8_v1_target_transition(
                    inbox_code=inbox_code,
                    transform=transform,
                )
                self.assertEqual(carrier, (input_carrier + 1) % 16)
                self.assertEqual(symbol, transform[input_symbol])
                self.assertEqual(
                    p.gate8_v1_decode_message_code(code),
                    (carrier, symbol),
                )
                outputs.add(code)
            self.assertEqual(outputs, set(range(256)))

    def test_root_symbol_enters_as_initial_low_nibble(self) -> None:
        transform = w.GATE8_TRANSFORM_PERMUTATIONS[0]
        for root_symbol in range(16):
            initial_code = p.gate8_v1_encode_message_code(
                carrier=0,
                symbol=root_symbol,
            )
            self.assertEqual(initial_code, root_symbol)
            carrier, symbol, code = p.gate8_v1_target_transition(
                inbox_code=initial_code,
                transform=transform,
            )
            self.assertEqual(carrier, 1)
            self.assertEqual(symbol, transform[root_symbol])
            self.assertEqual(code, 16 + transform[root_symbol])

    def test_invalid_codes_and_transforms_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "message code"):
            p.gate8_v1_decode_message_code(256)
        with self.assertRaisesRegex(ValueError, "carrier"):
            p.gate8_v1_encode_message_code(carrier=16, symbol=0)
        with self.assertRaisesRegex(ValueError, "exactly 16"):
            p.gate8_v1_target_transition(
                inbox_code=0,
                transform=tuple(range(15)),
            )
        with self.assertRaisesRegex(ValueError, "permutation"):
            p.gate8_v1_target_transition(
                inbox_code=0,
                transform=(0,) * 16,
            )

    def test_training_schedule_is_exact_and_identical_to_v0_control(self) -> None:
        self.assertEqual(p.GATE8_V1_TRAINING_WORLDS_PER_SEED, 262_144)
        self.assertEqual(p.GATE8_V1_TRAINING_WORLD_BATCH_SIZE, 256)
        self.assertEqual(p.GATE8_V1_OPTIMIZER_STEPS, 1_024)
        observed_first_cycle = tuple(
            (
                p.gate8_v1_training_world_address(index).population,
                p.gate8_v1_training_world_address(index).depth,
                p.gate8_v1_training_world_address(index).condition_world_index,
            )
            for index in range(6)
        )
        self.assertEqual(
            observed_first_cycle,
            tuple(
                (population, depth, 0)
                for population, depth in p.GATE8_V1_TRAINING_CONDITIONS
            ),
        )
        counts = p.gate8_v1_condition_world_counts()
        self.assertEqual(
            tuple(counts.values()),
            (43_691, 43_691, 43_691, 43_691, 43_690, 43_690),
        )
        self.assertEqual(sum(counts.values()), 262_144)

    def test_validation_range_is_fresh_and_exact(self) -> None:
        indices = p.gate8_v1_validation_world_indices()
        self.assertEqual(len(indices), 512)
        self.assertEqual(indices[0], 512)
        self.assertEqual(indices[-1], 1_023)
        self.assertEqual(len(set(indices)), 512)
        self.assertTrue(set(indices).isdisjoint(range(512)))

    def test_learning_rate_schedule_preserves_controlled_comparison(self) -> None:
        self.assertAlmostEqual(
            p.gate8_v1_learning_rate(1),
            p.GATE8_V1_LEARNING_RATE / p.GATE8_V1_WARMUP_STEPS,
        )
        self.assertEqual(
            p.gate8_v1_learning_rate(p.GATE8_V1_WARMUP_STEPS),
            p.GATE8_V1_LEARNING_RATE,
        )
        self.assertAlmostEqual(
            p.gate8_v1_learning_rate(p.GATE8_V1_OPTIMIZER_STEPS),
            p.GATE8_V1_MINIMUM_LEARNING_RATE,
        )
        self.assertGreater(
            p.gate8_v1_learning_rate(512),
            p.gate8_v1_learning_rate(768),
        )
        with self.assertRaisesRegex(ValueError, "optimizer step"):
            p.gate8_v1_learning_rate(0)

    def test_checkpoint_selection_uses_frozen_lexicographic_order(self) -> None:
        candidates = (
            _candidate(step=256, target=0.98, loss=0.01),
            _candidate(
                step=512,
                target=0.99,
                exact_message=0.998,
                symbol=0.999,
                loss=0.2,
            ),
            _candidate(
                step=768,
                target=0.99,
                exact_message=0.999,
                symbol=0.998,
                loss=0.3,
            ),
            _candidate(
                step=1_024,
                target=0.99,
                exact_message=0.999,
                symbol=0.999,
                loss=0.1,
            ),
        )
        self.assertEqual(
            p.select_gate8_v1_checkpoint(candidates).step,
            1_024,
        )

        exact_tie = tuple(
            _candidate(step=step, target=1.0, loss=0.0)
            for step in p.GATE8_V1_CHECKPOINT_STEPS
        )
        self.assertEqual(
            p.select_gate8_v1_checkpoint(exact_tie).step,
            256,
        )

    def test_admission_requires_every_inclusive_frozen_guard(self) -> None:
        admitted = tuple(
            _candidate(
                step=step,
                target=0.99,
                exact_message=0.995,
                carrier=0.995,
                symbol=0.995,
            )
            for step in p.GATE8_V1_CHECKPOINT_STEPS
        )
        self.assertEqual(
            p.classify_gate8_v1_training(admitted),
            p.GATE8_V1_TRAINING_ADMITTED,
        )

        failures = (
            dict(target=0.989),
            dict(target=1.0, exact_message=0.994),
            dict(target=1.0, carrier=0.994),
            dict(target=1.0, symbol=0.994),
            dict(target=1.0, inbox_coverage=255),
            dict(target=1.0, target_coverage=255),
            dict(target=1.0, target_carrier_coverage=15),
            dict(target=1.0, target_symbol_coverage=15),
        )
        for failure in failures:
            with self.subTest(failure=failure):
                candidates = tuple(
                    _candidate(step=step, **failure)
                    for step in p.GATE8_V1_CHECKPOINT_STEPS
                )
                self.assertEqual(
                    p.classify_gate8_v1_training(candidates),
                    p.GATE8_V1_TRAINING_NOT_ADMITTED,
                )

    def test_protocol_plan_preserves_closed_scientific_boundary(self) -> None:
        plan = p.gate8_v1_training_protocol_plan()
        self.assertEqual(
            plan["architecture_head"],
            "c3ab64008c816fa1eb6f9d6f8f0a1a99ed195ec8",
        )
        self.assertEqual(
            plan["runtime_head"],
            "333d88ac4fc52f1651741fba224e0b4605feedd3",
        )
        self.assertEqual(plan["training_seeds"], [0, 1, 2])
        self.assertTrue(plan["seed0_first"])
        self.assertTrue(plan["seeds_1_and_2_blocked_until_seed0_admission"])
        self.assertEqual(plan["validation_world_indices"], [512, 1_023])
        self.assertTrue(plan["validation_disjoint_from_v0_indices_0_through_511"])
        self.assertEqual(
            plan["loss_weights"],
            {
                "carrier_cross_entropy": 1.0,
                "symbol_cross_entropy": 1.0,
            },
        )
        self.assertEqual(
            plan["removed_losses"],
            [
                "joint_256_way_message_cross_entropy",
                "answer_cross_entropy",
                "activity_binary_cross_entropy",
            ],
        )
        self.assertTrue(
            all(depth <= 16 for _population, depth in p.GATE8_V1_TRAINING_CONDITIONS)
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
            "torch.load",
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
