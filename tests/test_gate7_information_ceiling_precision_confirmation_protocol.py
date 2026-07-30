from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest


PROTOCOL_PATH = pathlib.Path(
    "ai_hypothesis/population_compute/"
    "gate7_information_ceiling_precision_confirmation_protocol.py"
)


def _load_protocol():
    name = "gate7_information_ceiling_precision_confirmation_protocol_test_module"
    spec = importlib.util.spec_from_file_location(name, PROTOCOL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load precision-confirmation protocol")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


p = _load_protocol()


class Gate7InformationCeilingPrecisionProtocolTests(unittest.TestCase):
    def _cells(
        self,
        *,
        delta: float = -0.01,
        low: float = -0.015,
        high: float = -0.005,
        clear_gap_at: tuple[int, int] | None = None,
    ):
        rows = []
        for population in p.GATE7_PRECISION_POPULATIONS:
            for checkpoint in p.GATE7_PRECISION_CHECKPOINT_INDICES:
                row_low = low
                row_high = high
                row_delta = delta
                if clear_gap_at == (population, checkpoint):
                    row_low = -0.04
                    row_high = -0.021
                    row_delta = -0.03
                rows.append(
                    p.Gate7PrecisionCellComparison(
                        population=population,
                        checkpoint_index=checkpoint,
                        learned_minus_bayes_delta=row_delta,
                        learned_minus_bayes_ci_low=row_low,
                        learned_minus_bayes_ci_high=row_high,
                        learned_minus_hash_ci_low=0.04,
                        bayes_minus_hash_ci_low=0.05,
                    )
                )
        return tuple(rows)

    def _populations(
        self,
        *,
        delta: float = -0.01,
        low: float = -0.014,
        high: float = -0.006,
        clear_gap_population: int | None = None,
    ):
        rows = []
        for population in p.GATE7_PRECISION_POPULATIONS:
            row_low = low
            row_high = high
            row_delta = delta
            if population == clear_gap_population:
                row_low = -0.035
                row_high = -0.021
                row_delta = -0.028
            rows.append(
                p.Gate7PrecisionPopulationComparison(
                    population=population,
                    learned_minus_bayes_delta=row_delta,
                    learned_minus_bayes_ci_low=row_low,
                    learned_minus_bayes_ci_high=row_high,
                    learned_minus_hash_ci_low=0.04,
                    bayes_minus_hash_ci_low=0.05,
                )
            )
        return tuple(rows)

    def _pooled(self, *, delta: float, low: float, high: float):
        return p.Gate7PrecisionPooledComparison(
            learned_minus_bayes_delta=delta,
            learned_minus_bayes_ci_low=low,
            learned_minus_bayes_ci_high=high,
            learned_minus_hash_ci_low=0.04,
            bayes_minus_hash_ci_low=0.05,
        )

    def test_plan_freezes_exact_precision_matrix(self) -> None:
        plan = p.gate7_precision_confirmation_plan()
        self.assertFalse(plan["execution_admitted"])
        self.assertFalse(plan["training_performed"])
        self.assertFalse(plan["communication_intervention_performed"])
        self.assertFalse(plan["prior_worlds_reused"])
        self.assertEqual(plan["populations"], [16_384, 32_768, 65_536, 131_072])
        self.assertEqual(plan["checkpoint_indices"], [0, 1, 2])
        self.assertEqual(plan["world_count_per_checkpoint_population"], 2_048)
        self.assertEqual(plan["evaluation_batch_size"], 64)
        self.assertEqual(plan["physical_batch_count"], 32)
        self.assertEqual(plan["bootstrap_samples"], 20_000)
        self.assertEqual(plan["primary_attempts"], 128)
        self.assertEqual(plan["learned_parameter_count"], 19_649)
        self.assertEqual(
            plan["bootstrap_unit"],
            "world_index_clustered_within_population_across_T0_T1_T2",
        )
        self.assertEqual(plan["pooled_weighting"], "equal_population_then_equal_checkpoint")

    def test_pooled_noninferiority_without_local_gap_is_ceiling_dominant(self) -> None:
        outcome = p.classify_gate7_precision_confirmation(
            cells=self._cells(),
            populations=self._populations(),
            pooled=self._pooled(delta=-0.009, low=-0.012, high=-0.006),
        )
        self.assertEqual(outcome, p.GATE7_PRECISION_INFORMATION_CEILING_DOMINANT)

    def test_pooled_clear_gap_is_scorer_gap(self) -> None:
        outcome = p.classify_gate7_precision_confirmation(
            cells=self._cells(delta=-0.03, low=-0.04, high=-0.021),
            populations=self._populations(delta=-0.03, low=-0.04, high=-0.021),
            pooled=self._pooled(delta=-0.03, low=-0.035, high=-0.025),
        )
        self.assertEqual(outcome, p.GATE7_PRECISION_SCORER_REPRESENTATION_GAP)

    def test_local_clear_gap_with_no_pooled_clear_gap_is_mixed(self) -> None:
        outcome = p.classify_gate7_precision_confirmation(
            cells=self._cells(clear_gap_at=(65_536, 1)),
            populations=self._populations(clear_gap_population=65_536),
            pooled=self._pooled(delta=-0.012, low=-0.019, high=-0.006),
        )
        self.assertEqual(outcome, p.GATE7_PRECISION_INFORMATION_AND_SCORER_GAP_MIXED)

    def test_crossing_pooled_margin_without_local_clear_gap_is_inconclusive(self) -> None:
        outcome = p.classify_gate7_precision_confirmation(
            cells=self._cells(low=-0.025, high=-0.002),
            populations=self._populations(low=-0.024, high=-0.003),
            pooled=self._pooled(delta=-0.012, low=-0.023, high=-0.004),
        )
        self.assertEqual(outcome, p.GATE7_PRECISION_INCONCLUSIVE)

    def test_population_point_outside_margin_blocks_dominant_label(self) -> None:
        populations = list(self._populations())
        populations[2] = p.Gate7PrecisionPopulationComparison(
            population=65_536,
            learned_minus_bayes_delta=-0.021,
            learned_minus_bayes_ci_low=-0.03,
            learned_minus_bayes_ci_high=-0.005,
            learned_minus_hash_ci_low=0.04,
            bayes_minus_hash_ci_low=0.05,
        )
        outcome = p.classify_gate7_precision_confirmation(
            cells=self._cells(),
            populations=tuple(populations),
            pooled=self._pooled(delta=-0.009, low=-0.012, high=-0.006),
        )
        self.assertEqual(outcome, p.GATE7_PRECISION_INCONCLUSIVE)

    def test_incomplete_or_reordered_matrix_is_rejected(self) -> None:
        cells = self._cells()
        with self.assertRaisesRegex(ValueError, "population-major"):
            p.classify_gate7_precision_confirmation(
                cells=tuple(reversed(cells)),
                populations=self._populations(),
                pooled=self._pooled(delta=-0.009, low=-0.012, high=-0.006),
            )

    def test_protocol_file_has_no_execution_or_artifact_surface(self) -> None:
        text = PROTOCOL_PATH.read_text(encoding="utf-8")
        forbidden = (
            "import torch",
            "torch.",
            "open(",
            "read_text(",
            "json.load",
            "checkpoint_path",
            "world_generator",
            "execution_admitted\": True",
            "communication_intervention_performed\": True",
        )
        for token in forbidden:
            self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main()
