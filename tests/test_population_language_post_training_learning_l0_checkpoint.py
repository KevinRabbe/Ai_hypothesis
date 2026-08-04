from __future__ import annotations

from collections import OrderedDict
import hashlib
import pathlib
import tempfile
import unittest

import torch

from ai_hypothesis.population_language.l0_models import PopulationLanguageOrganism
from ai_hypothesis.population_language.l0_reference_training import (
    OPTIMIZER_STEPS,
    POPULATION_COMMUNICATION_ROUNDS,
    POPULATION_TOP_K,
    VERSION as REFERENCE_TRAINING_VERSION,
    canonical_state_sha256,
)
from ai_hypothesis.population_language.post_training_learning_l0_checkpoint import (
    REFERENCE_CHECKPOINT_MAX_BYTES,
    REFERENCE_CHECKPOINT_PAYLOAD_KEYS,
    REFERENCE_MODEL_NAME,
    REFERENCE_RAW_STATE_BYTES,
    decode_reference_checkpoint,
    load_reference_checkpoint,
    materialize_population_checkpoint_state,
    validate_checkpoint_contract,
)


class ReferenceCheckpointContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = pathlib.Path(cls.temporary.name)
        cls.checkpoint_path = cls.root / "population-seed-120100.pt"
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(120100)
            model = PopulationLanguageOrganism(
                communication_rounds=POPULATION_COMMUNICATION_ROUNDS,
                top_k=POPULATION_TOP_K,
            )
        cls.state = OrderedDict(
            (name, value.detach().cpu().contiguous())
            for name, value in model.state_dict().items()
        )
        cls.canonical_sha256 = canonical_state_sha256(model)
        torch.save(
            {
                "version": REFERENCE_TRAINING_VERSION,
                "model": REFERENCE_MODEL_NAME,
                "seed": 120100,
                "optimizer_step": OPTIMIZER_STEPS,
                "state_dict": cls.state,
            },
            cls.checkpoint_path,
        )
        cls.payload = cls.checkpoint_path.read_bytes()
        cls.file_sha256 = hashlib.sha256(cls.payload).hexdigest()
        del model

    @classmethod
    def tearDownClass(cls) -> None:
        cls.state.clear()
        cls.temporary.cleanup()

    def test_static_contract_is_exact_and_bounded(self) -> None:
        report = validate_checkpoint_contract()
        self.assertTrue(report["valid"])
        self.assertEqual(
            REFERENCE_CHECKPOINT_PAYLOAD_KEYS,
            ("version", "model", "seed", "optimizer_step", "state_dict"),
        )
        self.assertEqual(REFERENCE_CHECKPOINT_MAX_BYTES, 96 * 1024 * 1024)
        self.assertEqual(REFERENCE_RAW_STATE_BYTES, 18_967_968 * 4)
        self.assertLess(len(self.payload), REFERENCE_CHECKPOINT_MAX_BYTES)

    def test_exact_hash_pinned_checkpoint_round_trip(self) -> None:
        loaded = load_reference_checkpoint(
            self.checkpoint_path,
            expected_seed=120100,
            expected_file_sha256=self.file_sha256,
            expected_canonical_sha256=self.canonical_sha256,
        )
        self.assertEqual(loaded.seed, 120100)
        self.assertEqual(loaded.file_bytes, len(self.payload))
        self.assertEqual(loaded.file_sha256, self.file_sha256)
        self.assertEqual(loaded.canonical_state_sha256, self.canonical_sha256)
        self.assertFalse(loaded.model.training)
        self.assertEqual(canonical_state_sha256(loaded.model), self.canonical_sha256)
        self.assertTrue(all(parameter.device.type == "cpu" for parameter in loaded.model.parameters()))

    def test_state_contract_rejects_names_types_values_and_hash_drift(self) -> None:
        missing = OrderedDict(tuple(self.state.items())[:-1])
        with self.assertRaises(ValueError):
            materialize_population_checkpoint_state(
                missing,
                expected_canonical_sha256=self.canonical_sha256,
            )

        extra = OrderedDict(self.state)
        extra["unexpected"] = torch.zeros(1)
        with self.assertRaises(ValueError):
            materialize_population_checkpoint_state(
                extra,
                expected_canonical_sha256=self.canonical_sha256,
            )

        reordered = OrderedDict(reversed(tuple(self.state.items())))
        with self.assertRaises(ValueError):
            materialize_population_checkpoint_state(
                reordered,
                expected_canonical_sha256=self.canonical_sha256,
            )

        wrong_dtype = OrderedDict(self.state)
        wrong_dtype["lm_bias"] = wrong_dtype["lm_bias"].to(torch.float64)
        with self.assertRaises(ValueError):
            materialize_population_checkpoint_state(
                wrong_dtype,
                expected_canonical_sha256=self.canonical_sha256,
            )

        nonfinite = OrderedDict(self.state)
        nonfinite_bias = nonfinite["lm_bias"].clone()
        nonfinite_bias[0] = float("nan")
        nonfinite["lm_bias"] = nonfinite_bias
        with self.assertRaises(ValueError):
            materialize_population_checkpoint_state(
                nonfinite,
                expected_canonical_sha256=self.canonical_sha256,
            )

        with self.assertRaises(ValueError):
            materialize_population_checkpoint_state(
                self.state,
                expected_canonical_sha256="0" * 64,
            )

    def test_file_and_payload_boundaries_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            load_reference_checkpoint(
                self.checkpoint_path,
                expected_seed=120100,
                expected_file_sha256="0" * 64,
                expected_canonical_sha256=self.canonical_sha256,
            )
        with self.assertRaises(ValueError):
            load_reference_checkpoint(
                self.checkpoint_path,
                expected_seed=120101,
                expected_file_sha256=self.file_sha256,
                expected_canonical_sha256=self.canonical_sha256,
            )
        with self.assertRaises(TypeError):
            load_reference_checkpoint(  # type: ignore[arg-type]
                str(self.checkpoint_path),
                expected_seed=120100,
                expected_file_sha256=self.file_sha256,
                expected_canonical_sha256=self.canonical_sha256,
            )
        with self.assertRaises(ValueError):
            load_reference_checkpoint(
                self.root,
                expected_seed=120100,
                expected_file_sha256=self.file_sha256,
                expected_canonical_sha256=self.canonical_sha256,
            )
        with self.assertRaises(ValueError):
            decode_reference_checkpoint(
                b"not-a-torch-checkpoint",
                expected_seed=120100,
                expected_file_sha256=hashlib.sha256(b"not-a-torch-checkpoint").hexdigest(),
                expected_canonical_sha256=self.canonical_sha256,
            )

        wrong_metadata_path = self.root / "wrong-metadata.pt"
        torch.save(
            {
                "version": REFERENCE_TRAINING_VERSION,
                "model": "transformer",
                "seed": 120100,
                "optimizer_step": OPTIMIZER_STEPS,
                "state_dict": {},
            },
            wrong_metadata_path,
        )
        wrong_metadata = wrong_metadata_path.read_bytes()
        with self.assertRaises(ValueError):
            decode_reference_checkpoint(
                wrong_metadata,
                expected_seed=120100,
                expected_file_sha256=hashlib.sha256(wrong_metadata).hexdigest(),
                expected_canonical_sha256=self.canonical_sha256,
            )

    def test_model_construction_preserves_caller_rng_state(self) -> None:
        torch.manual_seed(9917)
        before = torch.random.get_rng_state().clone()
        model = materialize_population_checkpoint_state(
            self.state,
            expected_canonical_sha256=self.canonical_sha256,
        )
        after = torch.random.get_rng_state()
        self.assertTrue(torch.equal(before, after))
        del model


if __name__ == "__main__":
    unittest.main()
