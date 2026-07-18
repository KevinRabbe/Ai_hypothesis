from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path

from ai_hypothesis.dashboard.artifacts import normalize_artifact_ref
from ai_hypothesis.dashboard.errors import ResultsDirectoryAccessError
from ai_hypothesis.dashboard.indexer import DashboardIndexer, DashboardIndexSnapshot
from ai_hypothesis.dashboard.store import DashboardStore
from tests.test_dashboard_step01_adapter import step01_payload
from tests.test_dashboard_step02_adapter import step02_payload


class DashboardIndexerTests(unittest.TestCase):
    def test_absent_and_empty_results_directories_are_valid_states(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            absent = root / "missing"
            absent_snapshot = DashboardIndexer().build(absent)
            self.assertEqual(absent_snapshot.status.result_directory_status, "ABSENT")
            self.assertEqual(absent_snapshot.status.indexed_experiment_count, 0)

            empty = root / "empty"
            empty.mkdir()
            empty_snapshot = DashboardIndexer().build(empty)
            self.assertEqual(empty_snapshot.status.result_directory_status, "EMPTY")
            self.assertEqual(empty_snapshot.status.indexed_experiment_count, 0)

    def test_valid_and_malformed_artifacts_coexist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "good").mkdir()
            (root / "good" / "result.json").write_text(
                json.dumps(step01_payload()),
                encoding="utf-8",
            )
            (root / "bad").mkdir()
            (root / "bad" / "result.json").write_text("{not json", encoding="utf-8")

            snapshot = DashboardIndexer().build(root)

            self.assertEqual(snapshot.status.indexed_experiment_count, 1)
            self.assertEqual(snapshot.status.indexing_error_count, 1)

    def test_step1_and_step2_can_coexist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "step01").mkdir()
            (root / "step01" / "result.json").write_text(
                json.dumps(step01_payload()),
                encoding="utf-8",
            )
            (root / "step02").mkdir()
            (root / "step02" / "result.json").write_text(
                json.dumps(step02_payload()),
                encoding="utf-8",
            )

            snapshot = DashboardIndexer().build(root)

            self.assertEqual(snapshot.status.indexed_experiment_count, 2)
            self.assertEqual(snapshot.status.population_experiment_count, 1)

    def test_summary_without_authoritative_seed_metadata_does_not_create_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "summary.json").write_text(
                json.dumps(
                    {
                        "execution_mode": "strictly_sequential",
                        "experiments": [
                            {
                                "experiment_name": "step01_checkpoint_50k_extended_15k",
                                "parameter_count": 50268,
                                "test_accuracy": 0.9,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            snapshot = DashboardIndexer().build(root)

            self.assertEqual(snapshot.status.group_count, 0)

    def test_summary_with_expected_seeds_creates_complete_groups(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "summary.json").write_text(
                json.dumps(
                    {
                        "execution_mode": "strictly_sequential",
                        "requested_seeds": [1, 2],
                        "runs": [
                            {
                                "size_label": "50k",
                                "seed": 1,
                                "test_accuracy": 0.9,
                            },
                            {
                                "size_label": "50k",
                                "seed": 2,
                                "test_accuracy": 0.91,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            snapshot = DashboardIndexer().build(root)

            self.assertEqual(snapshot.status.group_count, 1)
            self.assertEqual(snapshot.status.complete_group_count, 1)

    def test_identical_duplicates_are_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("a", "b"):
                (root / name).mkdir()
                (root / name / "result.json").write_text(
                    json.dumps(step01_payload()),
                    encoding="utf-8",
                )

            snapshot = DashboardIndexer().build(root)

            self.assertEqual(snapshot.status.indexed_experiment_count, 1)
            self.assertEqual(snapshot.experiments[0].provenance.duplicate_artifact_count, 1)

    def test_conflicting_duplicate_is_index_error(self) -> None:
        first = step01_payload()
        second = step01_payload()
        second["test"]["accuracy"] = 0.1
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, payload in (("a", first), ("b", second)):
                (root / name).mkdir()
                (root / name / "result.json").write_text(json.dumps(payload), encoding="utf-8")

            snapshot = DashboardIndexer().build(root)

            self.assertEqual(snapshot.status.indexed_experiment_count, 0)
            self.assertGreaterEqual(snapshot.status.indexing_error_count, 1)

    def test_reindex_preserves_previous_snapshot_on_catastrophic_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "result.json").write_text(json.dumps(step01_payload()), encoding="utf-8")
            store = DashboardStore(results_dir=root)
            self.assertEqual(store.snapshot().status.indexed_experiment_count, 1)

            class FailingIndexer:
                def build(self, results_dir: Path) -> DashboardIndexSnapshot:
                    raise RuntimeError("catastrophic failure")

            store.indexer = FailingIndexer()  # type: ignore[assignment]
            snapshot = store.reindex()

            self.assertEqual(snapshot.status.indexed_experiment_count, 1)
            self.assertEqual(store.snapshot().status.indexed_experiment_count, 1)

    def test_snapshot_remains_readable_while_reindex_builds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "result.json").write_text(json.dumps(step01_payload()), encoding="utf-8")
            store = DashboardStore(results_dir=root)
            original = store.snapshot()
            started = threading.Event()
            release = threading.Event()

            class BlockingIndexer:
                def build(self, results_dir: Path) -> DashboardIndexSnapshot:
                    started.set()
                    release.wait(timeout=5)
                    return original

            store.indexer = BlockingIndexer()  # type: ignore[assignment]
            thread = threading.Thread(target=store.reindex)
            thread.start()
            self.assertTrue(started.wait(timeout=5))
            self.assertIs(store.snapshot(), original)
            release.set()
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())

    def test_artifact_ref_rejects_paths_outside_results_dir(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root.parent / "outside.json"
            with self.assertRaises(ResultsDirectoryAccessError):
                normalize_artifact_ref(root, outside)


if __name__ == "__main__":
    unittest.main()
