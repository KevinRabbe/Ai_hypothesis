from __future__ import annotations

import importlib.util
import pathlib
import sys
import tempfile
import unittest


RUNNER_PATH = pathlib.Path(
    "scripts/train_gate8_factorized_organism_replication.py"
)
BASE_RUNNER_PATH = pathlib.Path(
    "scripts/train_gate8_factorized_organism.py"
)
WRAPPER_PATH = pathlib.Path(
    "scripts/train_gate8_factorized_organism_replication.ps1"
)


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load Gate8 replication module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


replication = _load(
    RUNNER_PATH,
    "gate8_v1_replication_execution_test_module",
)
base = _load(
    BASE_RUNNER_PATH,
    "gate8_v1_seed0_runner_test_module",
)


class Gate8V1ReplicationExecutionTests(unittest.TestCase):
    def test_replication_admits_exactly_seeds_one_and_two(self) -> None:
        self.assertEqual(replication.ALLOWED_REPLICATION_SEEDS, (1, 2))
        self.assertNotIn(0, replication.ALLOWED_REPLICATION_SEEDS)
        for forbidden in (-1, 0, 3):
            with self.assertRaisesRegex(ValueError, "seeds 1 and 2 only"):
                replication.train_gate8_v1_replication(
                    seed=forbidden,
                    output_root=pathlib.Path(tempfile.gettempdir())
                    / f"forbidden-gate8-seed-{forbidden}",
                    device_name="cuda",
                )

    def test_replication_binds_qualified_seed0_result_and_frozen_stack(self) -> None:
        self.assertEqual(
            replication.QUALIFIED_SEED0_RESULT_HEAD,
            "f259620f7d3beab2f886c76271c753e9ebf96dc9",
        )
        self.assertEqual(
            replication.REPLICATION_EXECUTION_VERSION,
            "gate8-factorized-message-training-replication-execution-v1",
        )
        self.assertEqual(
            base.PROTOCOL_HEAD,
            "a33dc123d090268a531d112251ea3ab53cb50062",
        )
        self.assertEqual(
            base.RUNTIME_HEAD,
            "333d88ac4fc52f1651741fba224e0b4605feedd3",
        )
        self.assertEqual(
            base.ARCHITECTURE_HEAD,
            "c3ab64008c816fa1eb6f9d6f8f0a1a99ed195ec8",
        )
        self.assertEqual(
            base.EXPERIMENT_VERSION,
            "gate8-factorized-message-training-execution-v1",
        )

    def test_replication_reuses_frozen_mechanics_without_new_scientific_surface(
        self,
    ) -> None:
        text = RUNNER_PATH.read_text(encoding="utf-8")
        required = (
            "protocol.GATE8_V1_OPTIMIZER_STEPS + 1",
            "protocol.GATE8_V1_TRAINING_WORLD_BATCH_SIZE",
            'split="train"',
            "base._build_validation_cache",
            "base._validate_candidate",
            "protocol.GATE8_V1_CHECKPOINT_STEPS",
            "protocol.select_gate8_v1_checkpoint",
            "protocol.classify_gate8_v1_training",
            "base._checkpoint_payload",
            "torch.optim.AdamW",
            "clip_grad_norm_",
            '"world_index_start": 512',
            '"world_index_end_inclusive": 1_023',
            '"scientific_test_worlds_generated": False',
            '"reference_tokenizer_loaded": False',
            '"reference_model_weights_loaded": False',
            '"reference_inference_performed": False',
            '"seeds_1_and_2_executed": True',
        )
        for token in required:
            self.assertIn(token, text)

        forbidden = (
            'split="test"',
            'split="demonstration"',
            "AutoModel",
            "AutoTokenizer",
            "from_pretrained",
            "snapshot_download",
            "model.safetensors",
        )
        for token in forbidden:
            self.assertNotIn(token, text)

    def test_replication_keeps_seed0_runner_immutable_and_separate(self) -> None:
        replication_text = RUNNER_PATH.read_text(encoding="utf-8")
        base_text = BASE_RUNNER_PATH.read_text(encoding="utf-8")
        self.assertIn("if seed != 0:", base_text)
        self.assertIn("choices=(0,)", base_text)
        self.assertIn(
            "if seed not in ALLOWED_REPLICATION_SEEDS:",
            replication_text,
        )
        self.assertIn(
            "choices=ALLOWED_REPLICATION_SEEDS",
            replication_text,
        )
        self.assertNotEqual(RUNNER_PATH, BASE_RUNNER_PATH)

    def test_wrapper_is_guarded_and_replication_only(self) -> None:
        text = WRAPPER_PATH.read_text(encoding="utf-8")
        self.assertIn("[ValidateSet(1, 2)]", text)
        self.assertIn(
            "agent/gate8-factorized-message-training-replication-execution-v1",
            text,
        )
        self.assertIn("GATE8_V1_REPLICATION_WRAPPER_SMOKE", text)
        self.assertIn(
            "scripts/train_gate8_factorized_organism_replication.py",
            text,
        )
        self.assertIn("Seed 0 rerun:      FORBIDDEN", text)
        self.assertIn("Scientific test:   FORBIDDEN", text)
        self.assertIn("Reference model:   FORBIDDEN", text)
        self.assertIn("validation_world_index_start = 512", text)
        self.assertIn("validation_world_index_end_inclusive = 1023", text)

    def test_artifact_contract_preserves_four_candidates_and_hash_manifest(
        self,
    ) -> None:
        runner = RUNNER_PATH.read_text(encoding="utf-8")
        wrapper = WRAPPER_PATH.read_text(encoding="utf-8")
        self.assertIn('"checkpoint_candidates": candidate_details', runner)
        self.assertIn('"selected-checkpoint.pt"', runner)
        self.assertIn('"selected_checkpoint": {', runner)
        self.assertIn("Get-FileHash -Algorithm SHA256", wrapper)
        self.assertIn('"manifest.sha256"', wrapper)
        self.assertIn(
            "@($result.checkpoint_candidates).Count -ne 4",
            wrapper,
        )

    def test_no_replication_execution_occurs_in_ci_source(self) -> None:
        workflow = pathlib.Path(
            ".github/workflows/"
            "gate8-factorized-message-training-replication-execution-v1-ci.yml"
        ).read_text(encoding="utf-8")
        self.assertNotIn(
            "python scripts/train_gate8_factorized_organism_replication.py",
            workflow,
        )
        self.assertIn("GATE8_V1_REPLICATION_WRAPPER_SMOKE", workflow)
        self.assertNotIn('split="train"', workflow)
        self.assertNotIn('split="validation"', workflow)
        self.assertNotIn('split="test"', workflow)


if __name__ == "__main__":
    unittest.main()
