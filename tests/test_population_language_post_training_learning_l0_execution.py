from __future__ import annotations

from collections import OrderedDict
import copy
import pathlib
import tempfile
import unittest

import torch

from ai_hypothesis.population_language import l0_protocol
from ai_hypothesis.population_language import (
    post_training_learning_l0_adapter as adapter,
)
from ai_hypothesis.population_language import (
    post_training_learning_l0_calibration as calibration,
)
from ai_hypothesis.population_language import (
    post_training_learning_l0_execution as execution,
)
from ai_hypothesis.population_language import (
    post_training_learning_l0_protocol as protocol,
)
from ai_hypothesis.population_language import (
    post_training_learning_l0_world as world,
)


def _state(rank: int = 1) -> OrderedDict[str, torch.Tensor]:
    values = OrderedDict(
        (
            ("operator_embedding_delta", torch.zeros(8, 512)),
            (
                "encoder_down",
                torch.arange(rank * 14_544, dtype=torch.float32).reshape(
                    rank, 14_544
                )
                / 100_000,
            ),
            ("encoder_up", torch.zeros(128, rank)),
            (
                "decoder_down",
                torch.arange(rank * 128, dtype=torch.float32).reshape(rank, 128)
                / 10_000,
            ),
            ("decoder_up", torch.zeros(14_544, rank)),
            ("value_logit_bias", torch.arange(16, dtype=torch.float32) / 100),
        )
    )
    return values


class _FakeAdaptedModel:
    def __init__(self, rank: int) -> None:
        self._parameters = OrderedDict(
            (name, torch.nn.Parameter(value.clone()))
            for name, value in _state(rank).items()
        )

    def declared_adaptation_parameters(
        self,
    ) -> OrderedDict[str, torch.nn.Parameter]:
        return self._parameters


class PopulationLanguagePostTrainingLearningL0ExecutionTests(unittest.TestCase):
    def test_learning_example_encoding_stops_at_answer_marker(self) -> None:
        adaptation = world.make_example(
            "adaptation", 0, world.CALIBRATION_WORLD_SEEDS[0]
        )
        encoded = execution.encode_learning_example(adaptation)
        expected_prefix = adaptation.tokens[:-2]
        self.assertEqual(
            encoded.input_ids.tolist(),
            [l0_protocol.TOKEN_TO_ID[token] for token in expected_prefix],
        )
        self.assertEqual(
            encoded.input_ids[-1].item(),
            l0_protocol.TOKEN_TO_ID["<answer>"],
        )
        self.assertEqual(
            encoded.target_id,
            l0_protocol.TOKEN_TO_ID[adaptation.tokens[-2]],
        )
        self.assertNotEqual(
            len(encoded.input_ids),
            len(adaptation.tokens),
        )

        composition = world.make_example(
            "calibration", 0, world.CALIBRATION_WORLD_SEEDS[0]
        )
        encoded_composition = execution.encode_learning_example(composition)
        self.assertEqual(encoded_composition.input_ids.numel(), 6)

    def test_adaptation_schedule_is_exact_and_locked(self) -> None:
        self.assertEqual(
            execution.adaptation_microbatch_ordinals(0),
            tuple(range(8)),
        )
        self.assertEqual(
            execution.adaptation_microbatch_ordinals(8),
            tuple(range(8)),
        )
        for updates, expected_hash in execution.SCHEDULE_SHA256_BY_UPDATES.items():
            self.assertEqual(
                execution.adaptation_schedule_sha256(updates),
                expected_hash,
            )
            flattened = [
                ordinal
                for update in range(updates)
                for ordinal in execution.adaptation_microbatch_ordinals(update)
            ]
            self.assertEqual(
                len(flattened),
                updates * calibration.MICROBATCH_SIZE,
            )
            self.assertEqual(set(flattened), set(range(world.ADAPTATION_EXAMPLES)))

    def test_locked_optimizer_matches_candidate(self) -> None:
        candidate = calibration.CalibrationCandidate(1, 0.003, 64)
        model = _FakeAdaptedModel(rank=1)
        optimizer = execution.build_locked_optimizer(  # type: ignore[arg-type]
            model, candidate
        )
        group = optimizer.param_groups[0]
        self.assertEqual(group["lr"], 0.003)
        self.assertEqual(group["betas"], calibration.ADAMW_BETAS)
        self.assertEqual(group["eps"], calibration.ADAMW_EPSILON)
        self.assertEqual(group["weight_decay"], 0.0)

        with self.assertRaises(ValueError):
            execution.build_locked_optimizer(  # type: ignore[arg-type]
                model,
                calibration.CalibrationCandidate(2, 0.003, 64),
            )

    def test_tensor_only_artifact_round_trip_and_create_once(self) -> None:
        state = _state(rank=1)
        payload = execution.encode_adaptation_artifact(state)
        self.assertLessEqual(len(payload), protocol.MAX_PERSISTED_ADAPTATION_BYTES)
        decoded = execution.decode_adaptation_artifact(payload)
        self.assertEqual(tuple(decoded), adapter.NAMES)
        for name in adapter.NAMES:
            self.assertTrue(torch.equal(decoded[name], state[name]))

        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "adapter.ptl0"
            record = execution.save_adaptation_artifact_create_once(path, state)
            self.assertEqual(record.encoded_bytes, len(payload))
            self.assertEqual(record.raw_tensor_bytes, adapter.raw_fp32_bytes(1))
            loaded = execution.load_adaptation_artifact(
                path,
                expected_sha256=record.sha256,
            )
            for name in adapter.NAMES:
                self.assertTrue(torch.equal(loaded[name], state[name]))
            with self.assertRaises(FileExistsError):
                execution.save_adaptation_artifact_create_once(path, state)
            with self.assertRaises(ValueError):
                execution.load_adaptation_artifact(
                    path,
                    expected_sha256="0" * 64,
                )

    def test_artifact_rejects_wrong_order_nonfinite_and_trailing_bytes(self) -> None:
        state = _state(rank=1)
        with self.assertRaises(ValueError):
            execution.validate_adaptation_state(
                OrderedDict(reversed(tuple(state.items())))
            )
        nonfinite = copy.deepcopy(state)
        nonfinite["value_logit_bias"][0] = float("nan")
        with self.assertRaises(ValueError):
            execution.encode_adaptation_artifact(nonfinite)
        payload = execution.encode_adaptation_artifact(state)
        with self.assertRaises(ValueError):
            execution.decode_adaptation_artifact(payload + b"x")

    def test_paired_bootstrap_is_deterministic_and_chunk_invariant(self) -> None:
        baseline = [1, 0, 1, 0, 1, 0, 0, 0]
        adapted = [1, 1, 1, 1, 1, 1, 0, 1]
        seed = world.FINAL_WORLD_SEEDS[0]
        one = execution.paired_bootstrap_lower_bound(
            baseline,
            adapted,
            world_seed=seed,
            chunk_size=1,
        )
        seven = execution.paired_bootstrap_lower_bound(
            baseline,
            adapted,
            world_seed=seed,
            chunk_size=7,
        )
        large = execution.paired_bootstrap_lower_bound(
            baseline,
            adapted,
            world_seed=seed,
            chunk_size=20_000,
        )
        self.assertEqual(one, seven)
        self.assertEqual(one, large)
        self.assertEqual(one.seed, protocol.paired_bootstrap_seed(seed))
        self.assertEqual(one.resamples, 20_000)
        self.assertGreater(one.observed_mean_gain, 0.0)
        self.assertGreaterEqual(one.ci95_lower, 0.0)

    def test_execution_primitive_contract_is_locked(self) -> None:
        report = execution.validate_execution_primitives()
        self.assertTrue(report["valid"], report["checks"])
        self.assertEqual(
            execution.STATUS,
            "EXECUTION_PRIMITIVES_ONLY_NO_CHECKPOINT_OR_CALIBRATION_OR_FINAL_RESULT",
        )
        self.assertEqual(
            execution.SOURCE_CALIBRATION_HEAD,
            "19aa701c475b19fc5b31409528948f21ad9fbdf4",
        )


if __name__ == "__main__":
    unittest.main()
