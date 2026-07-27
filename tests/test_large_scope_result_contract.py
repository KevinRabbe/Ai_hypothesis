from __future__ import annotations

import copy
import unittest

from ai_hypothesis.large_scope.result_contract import validate_large_scope_result
from ai_hypothesis.large_scope.relevance import LARGE_SCOPE_BENCHMARK_VERSION


def _condition(mode: str, width: int, *, world_count: int = 1000) -> dict[str, object]:
    positive = world_count // 2
    negative = world_count - positive
    inspected = positive if width == 16 else positive * width // 16
    return {
        "split": "development",
        "mode": mode,
        "width": width,
        "world_count": world_count,
        "positive_world_count": positive,
        "negative_world_count": negative,
        "target_inspected_count": inspected,
        "target_retrieved_count": inspected // 2,
        "target_coverage_rate": inspected / positive,
        "target_retrieval_rate": (inspected // 2) / positive,
        "retrieval_given_inspected": (inspected // 2) / inspected if inspected else None,
        "mean_target_rank_when_inspected": 1.5 if inspected else None,
        "mean_target_relevant_evidence_when_inspected": 0.8 if inspected else None,
        "mean_strongest_distractor_relevant_evidence": 0.4,
        "mean_target_minus_distractor_evidence": 0.4 if inspected else None,
        "mean_candidate_relevant_evidence_positive": 0.7,
        "mean_candidate_relevant_evidence_negative": 0.3,
        "max_candidate_relevant_evidence_negative": 0.9,
    }


def _paired(width: int, *, world_count: int = 1000) -> dict[str, object]:
    positive = world_count // 2
    negative = world_count - positive
    inspected = positive if width == 16 else positive * width // 16
    identity = width == 1
    return {
        "split": "development",
        "width": width,
        "pair_count": world_count,
        "positive_world_count": positive,
        "negative_world_count": negative,
        "target_inspected_count": inspected,
        "same_target_retrieved_count": inspected // 2,
        "diverse_target_retrieved_count": inspected // 2,
        "both_retrieved_count": inspected // 2,
        "same_only_retrieved_count": 0,
        "diverse_only_retrieved_count": 0,
        "neither_retrieved_count": inspected - inspected // 2,
        "retrieval_given_inspected_same": (inspected // 2) / inspected if inspected else None,
        "retrieval_given_inspected_diverse": (inspected // 2) / inspected if inspected else None,
        "retrieval_given_inspected_delta": 0.0 if identity else 0.02,
        "retrieval_discordant_count": 0 if identity else 12,
        "exact_retrieval_discordance_p_value": None if identity else 0.21,
        "mean_target_rank_delta_when_inspected": 0.0 if identity else -0.1,
        "se_target_rank_delta_when_inspected": None if identity else 0.04,
        "mean_target_relevant_evidence_delta_when_inspected": 0.0 if identity else 0.03,
        "se_target_relevant_evidence_delta_when_inspected": None if identity else 0.01,
        "mean_strongest_distractor_relevant_evidence_delta": 0.0 if identity else -0.02,
        "se_strongest_distractor_relevant_evidence_delta": None if identity else 0.01,
        "mean_target_minus_distractor_gap_delta_when_inspected": 0.0 if identity else 0.05,
        "se_target_minus_distractor_gap_delta_when_inspected": None if identity else 0.02,
        "mean_candidate_relevant_evidence_positive_delta": 0.0 if identity else 0.01,
        "se_candidate_relevant_evidence_positive_delta": None if identity else 0.01,
        "mean_candidate_relevant_evidence_negative_delta": 0.0 if identity else -0.01,
        "se_candidate_relevant_evidence_negative_delta": None if identity else 0.01,
        "delta_definition": "diverse_workers_minus_same_worker",
    }


def valid_payload() -> dict[str, object]:
    widths = (1, 4, 16)
    checkpoints = [
        {
            "path": f"seed-{index}/best.pt",
            "step": 15000,
            "validation_score": 0.95,
            "checkpoint_id": f"weights-sha256-{index:064x}",
        }
        for index in range(1, 17)
    ]
    return {
        "benchmark_version": LARGE_SCOPE_BENCHMARK_VERSION,
        "split": "development",
        "world_count": 1000,
        "world_batch_size": 64,
        "start_seed": 0,
        "config": {
            "window_count": 16,
            "target_difficulty": "hard",
            "distractor_difficulty": "hard",
            "ambiguous_distractor_fraction": 0.125,
        },
        "widths": list(widths),
        "modes": ["same_worker", "diverse_workers"],
        "evidence_config": {},
        "population_width": 16,
        "unit_config": {},
        "checkpoints": checkpoints,
        "elapsed_seconds": 12.5,
        "local_window_evaluations": 1000 * 2 * sum(widths),
        "summaries": [
            _condition(mode, width)
            for mode in ("same_worker", "diverse_workers")
            for width in widths
        ],
        "paired_summaries": [_paired(width) for width in widths],
    }


class LargeScopeResultContractTests(unittest.TestCase):
    def test_valid_development_artifact_returns_normalized_readout(self) -> None:
        readout = validate_large_scope_result(valid_payload())
        self.assertEqual(readout.split, "development")
        self.assertEqual(readout.widths, (1, 4, 16))
        self.assertEqual(len(readout.checkpoint_ids), 16)
        self.assertEqual(len(readout.per_width), 3)
        self.assertEqual(readout.per_width[0].retrieval_given_inspected_delta, 0.0)
        self.assertEqual(readout.per_width[2].mean_target_minus_distractor_gap_delta, 0.05)

    def test_wrong_split_is_rejected(self) -> None:
        payload = valid_payload()
        payload["split"] = "test"
        with self.assertRaisesRegex(ValueError, "expected split"):
            validate_large_scope_result(payload)

    def test_duplicate_checkpoint_weight_identity_is_rejected(self) -> None:
        payload = valid_payload()
        checkpoints = payload["checkpoints"]
        assert isinstance(checkpoints, list)
        checkpoints[1]["checkpoint_id"] = checkpoints[0]["checkpoint_id"]
        with self.assertRaisesRegex(ValueError, "identities must be unique"):
            validate_large_scope_result(payload)

    def test_incomplete_condition_matrix_is_rejected(self) -> None:
        payload = valid_payload()
        summaries = payload["summaries"]
        assert isinstance(summaries, list)
        summaries.pop()
        with self.assertRaisesRegex(ValueError, "complete mode/width matrix"):
            validate_large_scope_result(payload)

    def test_same_and_diverse_target_coverage_must_match(self) -> None:
        payload = valid_payload()
        summaries = payload["summaries"]
        assert isinstance(summaries, list)
        for row in summaries:
            if row["mode"] == "diverse_workers" and row["width"] == 4:
                row["target_coverage_rate"] = 0.3
                break
        with self.assertRaisesRegex(ValueError, "deterministic target coverage"):
            validate_large_scope_result(payload)

    def test_width_one_nonzero_diversity_delta_is_rejected(self) -> None:
        payload = valid_payload()
        paired = payload["paired_summaries"]
        assert isinstance(paired, list)
        paired[0]["mean_target_relevant_evidence_delta_when_inspected"] = 1e-3
        with self.assertRaisesRegex(ValueError, "width-1 paired identity"):
            validate_large_scope_result(payload)

    def test_wrong_local_window_accounting_is_rejected(self) -> None:
        payload = valid_payload()
        payload["local_window_evaluations"] = 1
        with self.assertRaisesRegex(ValueError, "evaluation accounting"):
            validate_large_scope_result(payload)

    def test_pair_count_must_cover_every_world(self) -> None:
        payload = valid_payload()
        paired = payload["paired_summaries"]
        assert isinstance(paired, list)
        paired[2]["pair_count"] = 999
        with self.assertRaisesRegex(ValueError, "one pair per world"):
            validate_large_scope_result(payload)

    def test_numeric_tolerance_allows_tiny_width_one_roundoff(self) -> None:
        payload = valid_payload()
        paired = payload["paired_summaries"]
        assert isinstance(paired, list)
        paired[0]["mean_target_relevant_evidence_delta_when_inspected"] = 5e-8
        validate_large_scope_result(payload, numeric_tolerance=1e-7)
        with self.assertRaisesRegex(ValueError, "width-1 paired identity"):
            validate_large_scope_result(payload, numeric_tolerance=1e-9)

    def test_input_payload_is_not_mutated(self) -> None:
        payload = valid_payload()
        original = copy.deepcopy(payload)
        validate_large_scope_result(payload)
        self.assertEqual(payload, original)


if __name__ == "__main__":
    unittest.main()
