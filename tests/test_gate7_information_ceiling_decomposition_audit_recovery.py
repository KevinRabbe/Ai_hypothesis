from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_hypothesis.population_compute.analyze_gate7_information_ceiling_decomposition import (
    BAYES,
    HASH,
    LEARNED,
    RANKERS,
    Gate7InformationCeilingAudit,
)
from ai_hypothesis.population_compute.recover_gate7_information_ceiling_decomposition_audit import (
    RECOVERY_REASON,
    canonicalize_ranker_object_order,
    recover_gate7_information_ceiling_audit,
)


def _sorted_ranker_mapping() -> dict[str, list[int]]:
    return {
        BAYES: [2, 3],
        LEARNED: [1, 4],
        HASH: [5, 6],
    }


def _twelve_checkpoint_payload() -> dict[str, object]:
    return {
        "tiers": [
            {
                "population": population,
                "checkpoint_results": [
                    {
                        "checkpoint_index": checkpoint,
                        "ranks_by_ranker": _sorted_ranker_mapping(),
                    }
                    for checkpoint in range(3)
                ],
            }
            for population in (16_384, 32_768, 65_536, 131_072)
        ]
    }


class Gate7InformationCeilingAuditRecoveryTests(unittest.TestCase):
    def test_canonicalization_restores_semantic_ranker_order_only(self) -> None:
        payload = _twelve_checkpoint_payload()
        before_values = json.loads(json.dumps(payload, sort_keys=True))

        rewrites = canonicalize_ranker_object_order(payload)

        self.assertEqual(rewrites, 12)
        for tier in payload["tiers"]:
            for checkpoint in tier["checkpoint_results"]:
                self.assertEqual(tuple(checkpoint["ranks_by_ranker"]), RANKERS)
        self.assertEqual(payload, before_values)

    def test_recovery_round_trip_preserves_scientific_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "result.json"
            output = root / "audit.json"
            metadata = root / "recovery.json"
            payload = _twelve_checkpoint_payload()
            artifact.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            original = artifact.read_bytes()

            def fake_audit(path: Path) -> Gate7InformationCeilingAudit:
                canonical = json.loads(path.read_text(encoding="utf-8"))
                for tier in canonical["tiers"]:
                    for checkpoint in tier["checkpoint_results"]:
                        self.assertEqual(tuple(checkpoint["ranks_by_ranker"]), RANKERS)
                return Gate7InformationCeilingAudit(
                    artifact_valid=True,
                    scientific_status="FRESH_GATE7_INFORMATION_CEILING_DECOMPOSITION_EVIDENCE",
                    campaign_outcome="G7_INFORMATION_CEILING_INCONCLUSIVE",
                    primary_coverage_by_population_checkpoint={},
                    errors=(),
                )

            with patch(
                "ai_hypothesis.population_compute.recover_gate7_information_ceiling_decomposition_audit.audit_gate7_information_ceiling_decomposition",
                side_effect=fake_audit,
            ):
                status = recover_gate7_information_ceiling_audit(
                    artifact=artifact,
                    output=output,
                    metadata_output=metadata,
                )

            self.assertEqual(status, 0)
            self.assertEqual(artifact.read_bytes(), original)
            self.assertTrue(json.loads(output.read_text(encoding="utf-8"))["artifact_valid"])
            recovery = json.loads(metadata.read_text(encoding="utf-8"))
            self.assertEqual(recovery["reason"], RECOVERY_REASON)
            self.assertEqual(recovery["ranker_mappings_canonicalized"], 12)
            self.assertFalse(recovery["scientific_execution_repeated"])
            self.assertFalse(recovery["scientific_artifact_modified"])

    def test_recovery_rejects_incomplete_ranker_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "result.json"
            artifact.write_text(
                json.dumps({"tiers": []}, sort_keys=True),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "exactly 12"):
                recover_gate7_information_ceiling_audit(
                    artifact=artifact,
                    output=root / "audit.json",
                )


if __name__ == "__main__":
    unittest.main()
