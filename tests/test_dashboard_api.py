from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from ai_hypothesis.dashboard.app import create_app
from ai_hypothesis.dashboard.settings import make_settings
from tests.test_dashboard_step01_adapter import step01_payload


class DashboardApiTests(unittest.TestCase):
    def _client(self, results_dir: Path) -> TestClient:
        settings = make_settings(results_dir=str(results_dir))
        return TestClient(create_app(settings))

    def test_health_and_zero_data_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = self._client(Path(directory) / "absent")
            self.assertEqual(client.get("/api/v1/health").status_code, 200)
            status = client.get("/api/v1/status").json()["status"]
            self.assertEqual(status["indexed_experiment_count"], 0)
            self.assertEqual(status["result_directory_status"], "ABSENT")
            experiments = client.get("/api/v1/experiments").json()
            self.assertEqual(experiments["items"], [])

    def test_experiment_lookup_unknown_id_and_reindex(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.mkdir(exist_ok=True)
            client = self._client(root)
            self.assertEqual(client.get("/api/v1/experiments/nope").status_code, 404)

            path = root / "result.json"
            path.write_text(json.dumps(step01_payload()), encoding="utf-8")
            reindex_status = client.post("/api/v1/reindex").json()["status"]
            self.assertEqual(reindex_status["indexed_experiment_count"], 1)
            experiments = client.get("/api/v1/experiments").json()
            experiment_id = experiments["items"][0]["identity"]["experiment_id"]
            self.assertEqual(
                client.get(f"/api/v1/experiments/{experiment_id}").status_code,
                200,
            )

    def test_reindex_adds_valid_artifacts_isolates_malformed_and_does_not_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.mkdir(exist_ok=True)
            (root / "a").mkdir()
            (root / "a" / "result.json").write_text(
                json.dumps(step01_payload()),
                encoding="utf-8",
            )
            client = self._client(root)
            self.assertEqual(
                client.get("/api/v1/status").json()["status"]["indexed_experiment_count"],
                1,
            )

            payload = step01_payload()
            payload["train_config"]["seed"] = 2
            (root / "b").mkdir()
            (root / "b" / "result.json").write_text(json.dumps(payload), encoding="utf-8")
            (root / "bad").mkdir()
            (root / "bad" / "result.json").write_text("{bad json", encoding="utf-8")

            first = client.post("/api/v1/reindex").json()["status"]
            second = client.post("/api/v1/reindex").json()["status"]

            self.assertEqual(first["indexed_experiment_count"], 2)
            self.assertEqual(first["indexing_error_count"], 1)
            self.assertEqual(second["indexed_experiment_count"], 2)
            self.assertEqual(second["indexing_error_count"], 1)


if __name__ == "__main__":
    unittest.main()
