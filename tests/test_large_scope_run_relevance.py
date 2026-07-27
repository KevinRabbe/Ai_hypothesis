from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import torch

from ai_hypothesis.large_scope import run_relevance
from ai_hypothesis.step01.model import LABEL_TO_INDEX, NON_UNCERTAIN_LABELS, Step01Output


@dataclass(frozen=True)
class _FakeUnitConfig:
    hidden_dim: int = 16
    source: str = "synthetic-ci"


@dataclass(frozen=True)
class _FakeCheckpoint:
    path: str
    seed: int


class _FakeCheckpointBank:
    def __init__(self, population_width: int = 16) -> None:
        self._population_width = population_width
        self.device = torch.device("cpu")
        self.unit_config = _FakeUnitConfig()
        self.checkpoints = tuple(
            _FakeCheckpoint(path=f"fake-{index}.pt", seed=index)
            for index in range(population_width)
        )
        self.calls = 0

    @property
    def population_width(self) -> int:
        return self._population_width

    def forward_selected(self, worker_indices, features, mask):
        del mask
        self.calls += 1
        indices = torch.as_tensor(worker_indices, dtype=torch.float32, device=features.device)
        batch = int(features.shape[0])
        logits = torch.full(
            (batch, len(NON_UNCERTAIN_LABELS)),
            -6.0,
            dtype=features.dtype,
            device=features.device,
        )
        relevant = LABEL_TO_INDEX["RELEVANT"]
        not_relevant = LABEL_TO_INDEX["NOT_RELEVANT"]
        signal = features[:, :, 0].sum(dim=1) / 8.0 + indices * 0.001
        logits[:, relevant] = signal
        logits[:, not_relevant] = -signal
        return Step01Output(
            label_logits=logits,
            uncertainty_logits=torch.full(
                (batch,),
                -3.0,
                dtype=features.dtype,
                device=features.device,
            ),
        )


class LargeScopeRunRelevanceTests(unittest.TestCase):
    def test_test_split_is_refused_before_checkpoint_loading(self) -> None:
        with patch.object(
            run_relevance.HomogeneousWorkerBank,
            "from_checkpoints",
        ) as checkpoint_loader:
            with self.assertRaisesRegex(SystemExit, "Refusing to open the frozen test split"):
                run_relevance.main(
                    [
                        "--checkpoints",
                        "unused.pt",
                        "--split",
                        "test",
                    ]
                )
        checkpoint_loader.assert_not_called()

    def test_duplicate_modes_are_rejected_before_checkpoint_loading(self) -> None:
        with patch.object(
            run_relevance.HomogeneousWorkerBank,
            "from_checkpoints",
        ) as checkpoint_loader:
            with self.assertRaisesRegex(SystemExit, "--modes must be unique"):
                run_relevance.main(
                    [
                        "--checkpoints",
                        "unused.pt",
                        "--modes",
                        "same_worker",
                        "same_worker",
                    ]
                )
        checkpoint_loader.assert_not_called()

    def test_successful_cli_path_writes_threshold_free_result_artifact(self) -> None:
        bank = _FakeCheckpointBank()
        with tempfile.TemporaryDirectory() as tempdir:
            output = Path(tempdir) / "large-scope.json"
            with patch.object(
                run_relevance.HomogeneousWorkerBank,
                "from_checkpoints",
                return_value=bank,
            ):
                exit_code = run_relevance.main(
                    [
                        "--checkpoints",
                        "fake-0.pt",
                        "fake-1.pt",
                        "--device",
                        "cpu",
                        "--backend",
                        "loop",
                        "--split",
                        "development",
                        "--world-count",
                        "4",
                        "--world-batch-size",
                        "2",
                        "--window-count",
                        "4",
                        "--widths",
                        "1",
                        "4",
                        "--modes",
                        "same_worker",
                        "diverse_workers",
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue(output.is_file())
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["split"], "development")
            self.assertEqual(payload["world_count"], 4)
            self.assertEqual(payload["config"]["window_count"], 4)
            self.assertEqual(payload["widths"], [1, 4])
            self.assertEqual(payload["modes"], ["same_worker", "diverse_workers"])
            self.assertEqual(payload["population_width"], 16)
            self.assertEqual(payload["local_window_evaluations"], 40)
            self.assertEqual(len(payload["summaries"]), 4)
            self.assertEqual(len(payload["paired_summaries"]), 2)
            paired = {row["width"]: row for row in payload["paired_summaries"]}
            self.assertEqual(
                paired[1]["delta_definition"],
                "diverse_workers_minus_same_worker",
            )
            self.assertAlmostEqual(
                paired[1]["mean_candidate_relevant_evidence_positive_delta"],
                0.0,
                places=7,
            )
            self.assertAlmostEqual(
                paired[1]["mean_candidate_relevant_evidence_negative_delta"],
                0.0,
                places=7,
            )
            self.assertGreater(bank.calls, 0)
            self.assertNotIn("acceptance_threshold", payload)


if __name__ == "__main__":
    unittest.main()
