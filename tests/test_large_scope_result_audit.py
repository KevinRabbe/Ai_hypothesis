from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from ai_hypothesis.large_scope import audit_relevance
from ai_hypothesis.large_scope.result_audit import (
    audit_large_scope_result,
    render_large_scope_audit_markdown,
)


def _condition(
    mode: str,
    width: int,
    *,
    target_inspected_count: int,
    target_retrieved_count: int,
    retrieval_given_inspected: float,
) -> dict[str, object]:
    return {
        "split": "development",
        "mode": mode,
        "width": width,
        "world_count": 4,
        "positive_world_count": 2,
        "negative_world_count": 2,
        "target_inspected_count": target_inspected_count,
        "target_retrieved_count": target_retrieved_count,
        "target_coverage_rate": target_inspected_count / 2.0,
        "target_retrieval_rate": target_retrieved_count / 2.0,
        "retrieval_given_inspected": retrieval_given_inspected,
        "mean_target_rank_when_inspected": 1.0 if width == 1 else 1.5,
        "mean_target_relevant_evidence_when_inspected": 0.8,
        "mean_strongest_distractor_relevant_evidence": 0.4,
        "mean_target_minus_distractor_evidence": 0.4 if width > 1 else None,
        "mean_candidate_relevant_evidence_positive": 0.75,
        "mean_candidate_relevant_evidence_negative": 0.3,
        "max_candidate_relevant_evidence_negative": 0.5,
    }


def _paired(
    width: int,
    *,
    target_inspected_count: int,
    same_retrieved: int,
    diverse_retrieved: int,
    retrieval_same: float,
    retrieval_diverse: float,
) -> dict[str, object]:
    diverse_only = max(0, diverse_retrieved - same_retrieved)
    same_only = max(0, same_retrieved - diverse_retrieved)
    both = min(same_retrieved, diverse_retrieved)
    neither = target_inspected_count - both - diverse_only - same_only
    discordant = diverse_only + same_only
    return {
        "split": "development",
        "width": width,
        "pair_count": 4,
        "positive_world_count": 2,
        "negative_world_count": 2,
        "target_inspected_count": target_inspected_count,
        "same_target_retrieved_count": same_retrieved,
        "diverse_target_retrieved_count": diverse_retrieved,
        "both_retrieved_count": both,
        "same_only_retrieved_count": same_only,
        "diverse_only_retrieved_count": diverse_only,
        "neither_retrieved_count": neither,
        "retrieval_given_inspected_same": retrieval_same,
        "retrieval_given_inspected_diverse": retrieval_diverse,
        "retrieval_given_inspected_delta": retrieval_diverse - retrieval_same,
        "retrieval_discordant_count": discordant,
        "exact_retrieval_discordance_p_value": None if discordant == 0 else 1.0,
        "mean_target_rank_delta_when_inspected": 0.0 if width == 1 else -0.5,
        "se_target_rank_delta_when_inspected": 0.0 if width == 1 else 0.1,
        "mean_target_relevant_evidence_delta_when_inspected": 0.0 if width == 1 else 0.15,
        "se_target_relevant_evidence_delta_when_inspected": 0.0 if width == 1 else 0.05,
        "mean_strongest_distractor_relevant_evidence_delta": 0.0 if width == 1 else -0.1,
        "se_strongest_distractor_relevant_evidence_delta": 0.0 if width == 1 else 0.03,
        "mean_target_minus_distractor_gap_delta_when_inspected": None if width == 1 else 0.25,
        "se_target_minus_distractor_gap_delta_when_inspected": None if width == 1 else 0.06,
        "mean_candidate_relevant_evidence_positive_delta": 0.0 if width == 1 else 0.1,
        "se_candidate_relevant_evidence_positive_delta": 0.0 if width == 1 else 0.02,
        "mean_candidate_relevant_evidence_negative_delta": 0.0 if width == 1 else -0.05,
        "se_candidate_relevant_evidence_negative_delta": 0.0 if width == 1 else 0.01,
        "delta_definition": "diverse_workers_minus_same_worker",
    }


def valid_payload() -> dict[str, object]:
    return {
        "benchmark_version": "large-scope-relevance-v0",
        "split": "development",
        "world_count": 4,
        "world_batch_size": 2,
        "start_seed": 0,
        "config": {
            "window_count": 4,
            "target_difficulty": "hard",
            "distractor_difficulty": "hard",
            "ambiguous_distractor_fraction": 0.125,
        },
        "widths": [1, 4],
        "modes": ["same_worker", "diverse_workers"],
        "population_width": 16,
        "local_window_evaluations": 40,
        "summaries": [
            _condition(
                "same_worker",
                1,
                target_inspected_count=1,
                target_retrieved_count=1,
                retrieval_given_inspected=1.0,
            ),
            _condition(
                "diverse_workers",
                1,
                target_inspected_count=1,
                target_retrieved_count=1,
                retrieval_given_inspected=1.0,
            ),
            _condition(
                "same_worker",
                4,
                target_inspected_count=2,
                target_retrieved_count=1,
                retrieval_given_inspected=0.5,
            ),
            _condition(
                "diverse_workers",
                4,
                target_inspected_count=2,
                target_retrieved_count=2,
                retrieval_given_inspected=1.0,
            ),
        ],
        "paired_summaries": [
            _paired(
                1,
                target_inspected_count=1,
                same_retrieved=1,
                diverse_retrieved=1,
                retrieval_same=1.0,
                retrieval_diverse=1.0,
            ),
            _paired(
                4,
                target_inspected_count=2,
                same_retrieved=1,
                diverse_retrieved=2,
                retrieval_same=0.5,
                retrieval_diverse=1.0,
            ),
        ],
    }


class LargeScopeResultAuditTests(unittest.TestCase):
    def test_valid_result_passes_without_scientific_threshold(self) -> None:
        payload = valid_payload()
        audit = audit_large_scope_result(payload)
        self.assertTrue(audit.valid)
        self.assertEqual(audit.errors, ())
        markdown = render_large_scope_audit_markdown(payload, audit)
        self.assertIn("**Integrity:** VALID", markdown)
        self.assertIn("Paired diversity summaries", markdown)
        self.assertIn("This audit intentionally does **not** apply a research-success threshold.", markdown)

    def test_width1_nonzero_diversity_delta_is_integrity_failure(self) -> None:
        payload = valid_payload()
        payload["paired_summaries"][0]["mean_candidate_relevant_evidence_negative_delta"] = 0.01
        audit = audit_large_scope_result(payload)
        self.assertFalse(audit.valid)
        self.assertIn("WIDTH1_CONTROL_DELTA", {issue.code for issue in audit.errors})

    def test_scope_mismatch_between_modes_is_integrity_failure(self) -> None:
        payload = valid_payload()
        payload["summaries"][1]["target_inspected_count"] = 0
        payload["summaries"][1]["target_coverage_rate"] = 0.0
        audit = audit_large_scope_result(payload)
        codes = {issue.code for issue in audit.errors}
        self.assertIn("SCOPE_MODE_MISMATCH", codes)
        self.assertIn("SCOPE_COVERAGE_MISMATCH", codes)

    def test_local_window_accounting_must_close_exactly(self) -> None:
        payload = valid_payload()
        payload["local_window_evaluations"] = 39
        audit = audit_large_scope_result(payload)
        self.assertIn("LOCAL_EVALUATION_COUNT", {issue.code for issue in audit.errors})

    def test_missing_paired_width_is_integrity_failure_when_both_modes_exist(self) -> None:
        payload = valid_payload()
        payload["paired_summaries"] = payload["paired_summaries"][:1]
        audit = audit_large_scope_result(payload)
        self.assertIn("PAIRED_MISSING", {issue.code for issue in audit.errors})

    def test_test_split_is_locked_by_default(self) -> None:
        payload = valid_payload()
        payload["split"] = "test"
        for row in payload["summaries"]:
            row["split"] = "test"
        for row in payload["paired_summaries"]:
            row["split"] = "test"
        locked = audit_large_scope_result(payload)
        self.assertIn("TEST_SPLIT_LOCKED", {issue.code for issue in locked.errors})
        unlocked = audit_large_scope_result(payload, allow_test_split=True)
        self.assertTrue(unlocked.valid)

    def test_cli_writes_markdown_and_returns_nonzero_only_for_integrity_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            source = root / "development.json"
            output = root / "development.audit.md"
            source.write_text(json.dumps(valid_payload()), encoding="utf-8")
            self.assertEqual(
                audit_relevance.main(
                    ["--input", str(source), "--output", str(output)]
                ),
                0,
            )
            self.assertIn("**Integrity:** VALID", output.read_text(encoding="utf-8"))

            broken = valid_payload()
            broken["local_window_evaluations"] = 1
            source.write_text(json.dumps(broken), encoding="utf-8")
            self.assertEqual(
                audit_relevance.main(
                    ["--input", str(source), "--output", str(output)]
                ),
                2,
            )
            self.assertIn("LOCAL_EVALUATION_COUNT", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
