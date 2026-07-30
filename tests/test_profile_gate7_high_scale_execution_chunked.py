from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path

from ai_hypothesis.population_compute import (
    profile_gate7_high_scale_execution_chunked as profile,
)


class Gate7HighScaleChunkedEngineeringProfileTests(unittest.TestCase):
    def test_profile_freezes_chunked_v1_identity(self) -> None:
        self.assertTrue(profile.GATE7_HIGH_SCALE_CHUNKED_ENGINEERING_PROFILE_ONLY)
        self.assertEqual(
            profile.GATE7_HIGH_SCALE_CHUNKED_ENGINEERING_PROFILE_VERSION,
            "gate7-high-scale-execution-engineering-profile-chunked-v1",
        )
        self.assertEqual(profile.GATE7_HIGH_SCALE_FRONTIER_MAX_RECURRENT_ROWS, 1_048_576)
        self.assertEqual(profile.GATE7_HIGH_SCALE_EVALUATION_BATCH_SIZE, 64)

    def test_chunk_metadata_records_full_world_batch_and_tier_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(
                json.dumps(
                    {
                        "profile_version": "old",
                        "tiers": [
                            {"population": 1024},
                            {"population": 65536},
                            {"population": 131072},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            profile._add_chunk_metadata(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["profile_version"],
                "gate7-high-scale-execution-engineering-profile-chunked-v1",
            )
            self.assertTrue(payload["frontier_row_chunking_enabled"])
            self.assertEqual(payload["frontier_max_recurrent_rows"], 1_048_576)
            self.assertEqual(payload["world_batch_preserved"], 64)
            by_population = {row["population"]: row for row in payload["tiers"]}
            self.assertEqual(by_population[1024]["final_layer_recurrent_chunks_per_action"], 1)
            self.assertEqual(by_population[65536]["final_layer_recurrent_chunks_per_action"], 2)
            self.assertEqual(by_population[131072]["final_layer_recurrent_chunks_per_action"], 4)
            self.assertEqual(
                by_population[131072]["final_layer_parent_rows_per_action"],
                4_194_304,
            )

    def test_cli_exposes_only_output_root(self) -> None:
        source = inspect.getsource(profile.main)
        self.assertIn('"--output-root"', source)
        for forbidden in (
            "--population",
            "--batch",
            "--chunk",
            "--k",
            "--checkpoint",
            "--compiler",
            "--mixed-precision",
            "--world",
        ):
            self.assertNotIn(forbidden, source)

    def test_wrapper_module_has_no_scientific_execution_surface(self) -> None:
        source = inspect.getsource(profile)
        self.assertNotIn("torch.load", source)
        self.assertNotIn("load_verified", source)
        self.assertNotIn("checkpoint_path", source)
        self.assertNotIn("covered_by_world", source)
        self.assertNotIn("classify_gate7", source)
        self.assertNotIn("generate_gate7_high_scale_world", source)
        self.assertNotIn("hidden_path =", source)


if __name__ == "__main__":
    unittest.main()
