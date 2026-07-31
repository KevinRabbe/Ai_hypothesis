from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "ai_hypothesis/population_compute/gate8_seed0_causal_diagnostic_protocol.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("gate8_seed0_diagnostic_protocol", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Gate8 seed0 diagnostic protocol")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


P = _load()


class Gate8Seed0DiagnosticProtocolTests(unittest.TestCase):
    def _runtime_rows(
        self,
        *,
        baseline: float = 0.40,
        forced_active: float = 0.43,
        message_decode: float = 0.48,
        combined: float = 0.52,
    ):
        values = (baseline, forced_active, message_decode, combined)
        return tuple(
            P.Gate8RuntimeProbeMetrics(
                probe=probe,
                mean_target_accuracy=value,
                minimum_target_accuracy=value,
                condition_target_accuracies=(value,) * len(P.GATE8_VALIDATION_CONDITIONS),
            )
            for probe, value in zip(P.GATE8_RUNTIME_PROBES, values, strict=True)
        )

    def _training_row(
        self,
        *,
        probe: str,
        step: int,
        message: float,
        answer: float,
        activity: float,
        mean_target: float,
        message_invariance: float,
        answer_invariance: float,
    ):
        return P.Gate8TrainingProbeMetrics(
            probe=probe,
            step=step,
            message_accuracy=message,
            answer_accuracy=answer,
            activity_accuracy=activity,
            mean_target_accuracy=mean_target,
            minimum_target_accuracy=min(mean_target, 0.99),
            message_root_invariance=message_invariance,
            answer_root_invariance=answer_invariance,
        )

    def test_exact_checkpoint_and_result_bindings(self):
        self.assertEqual(P.GATE8_SEED, 0)
        self.assertEqual(P.GATE8_LEARNED_PARAMETER_COUNT, 19_649)
        self.assertEqual(
            P.GATE8_SEED0_RESULT_HEAD,
            "70e7e40149f9259d36b0e37ab17fc8c30370201e",
        )
        self.assertEqual(
            P.GATE8_SEED0_CHECKPOINT_SHA256,
            "4aca6bfde7fa82cd2c1fec3613c4cc59303788616f352e8d419c90662d7b9a1b",
        )
        self.assertEqual(
            P.GATE8_SEED0_RESULT_SHA256,
            "5f477022ac45a80d8b05b112b3485e4112519ddef78e0bb23990a090e0cc92e2",
        )

    def test_fresh_training_ranges_are_exact_and_disjoint(self):
        self.assertEqual(P.GATE8_HEAD_ONLY_WORLD_START, 262_144)
        self.assertEqual(P.GATE8_HEAD_ONLY_WORLD_END_EXCLUSIVE, 327_680)
        self.assertEqual(P.GATE8_FULL_RESUME_WORLD_START, 327_680)
        self.assertEqual(P.GATE8_FULL_RESUME_WORLD_END_EXCLUSIVE, 458_752)
        self.assertEqual(
            P.GATE8_HEAD_ONLY_WORLD_COUNT,
            P.GATE8_HEAD_ONLY_STEPS * P.GATE8_WORLD_BATCH_SIZE,
        )
        self.assertEqual(
            P.GATE8_FULL_RESUME_WORLD_COUNT,
            P.GATE8_FULL_RESUME_STEPS * P.GATE8_WORLD_BATCH_SIZE,
        )
        self.assertEqual(
            P.GATE8_HEAD_ONLY_WORLD_END_EXCLUSIVE,
            P.GATE8_FULL_RESUME_WORLD_START,
        )

    def test_runtime_probe_order_and_validation_surface_are_frozen(self):
        self.assertEqual(
            P.GATE8_RUNTIME_PROBES,
            (
                "baseline",
                "forced_active",
                "message_low4_decode",
                "forced_active_message_low4_decode",
            ),
        )
        self.assertEqual(len(P.GATE8_VALIDATION_CONDITIONS), 6)
        self.assertEqual(P.GATE8_VALIDATION_WORLDS_PER_CONDITION, 512)

    def test_classifier_can_report_mixed_findings(self):
        findings = P.gate8_classify_diagnostic(
            runtime_rows=self._runtime_rows(),
            head_only=self._training_row(
                probe="head_only",
                step=256,
                message=0.996,
                answer=0.991,
                activity=0.9995,
                mean_target=0.88,
                message_invariance=0.995,
                answer_invariance=0.995,
            ),
            full_resume=self._training_row(
                probe="full_resume",
                step=512,
                message=P.GATE8_BASELINE_MESSAGE_ACCURACY + 0.04,
                answer=0.90,
                activity=0.999,
                mean_target=P.GATE8_BASELINE_MEAN_TARGET_ACCURACY + 0.12,
                message_invariance=0.94,
                answer_invariance=0.96,
            ),
        )
        self.assertTrue(findings.activity_gate_material)
        self.assertTrue(findings.answer_head_material)
        self.assertTrue(findings.frozen_core_linearly_sufficient)
        self.assertTrue(findings.continued_optimization_effective)
        self.assertTrue(findings.core_interference_persists)

    def test_threshold_boundaries_are_inclusive(self):
        baseline = 0.40
        rows = self._runtime_rows(
            baseline=baseline,
            forced_active=baseline + P.GATE8_MATERIAL_RUNTIME_DELTA,
            message_decode=baseline + P.GATE8_MATERIAL_RUNTIME_DELTA,
            combined=baseline,
        )
        findings = P.gate8_classify_diagnostic(
            runtime_rows=rows,
            head_only=self._training_row(
                probe="head_only",
                step=256,
                message=P.GATE8_HEAD_ONLY_MESSAGE_SUFFICIENCY,
                answer=P.GATE8_HEAD_ONLY_ANSWER_SUFFICIENCY,
                activity=P.GATE8_HEAD_ONLY_ACTIVITY_SUFFICIENCY,
                mean_target=0.80,
                message_invariance=P.GATE8_HEAD_ONLY_ROOT_INVARIANCE,
                answer_invariance=P.GATE8_HEAD_ONLY_ROOT_INVARIANCE,
            ),
            full_resume=self._training_row(
                probe="full_resume",
                step=512,
                message=P.GATE8_BASELINE_MESSAGE_ACCURACY + P.GATE8_RESUME_MESSAGE_GAIN,
                answer=0.80,
                activity=0.995,
                mean_target=P.GATE8_BASELINE_MEAN_TARGET_ACCURACY
                + P.GATE8_RESUME_MEAN_TARGET_GAIN,
                message_invariance=P.GATE8_PERSISTENT_ROOT_INTERFERENCE,
                answer_invariance=P.GATE8_PERSISTENT_ROOT_INTERFERENCE,
            ),
        )
        self.assertTrue(findings.activity_gate_material)
        self.assertTrue(findings.answer_head_material)
        self.assertTrue(findings.frozen_core_linearly_sufficient)
        self.assertTrue(findings.continued_optimization_effective)
        self.assertFalse(findings.core_interference_persists)

    def test_protocol_plan_keeps_exposure_closed(self):
        plan = P.gate8_seed0_diagnostic_protocol_plan()
        self.assertFalse(plan["execution_admitted"])
        self.assertFalse(plan["scientific_test_worlds_admitted"])
        self.assertFalse(plan["seeds_1_2_admitted"])
        self.assertFalse(plan["reference_model_admitted"])
        self.assertEqual(plan["head_only"]["fresh_world_range"], [262_144, 327_680])
        self.assertEqual(plan["full_resume"]["fresh_world_range"], [327_680, 458_752])

    def test_invalid_rows_fail_closed(self):
        with self.assertRaises(ValueError):
            P.Gate8RuntimeProbeMetrics(
                probe="unknown",
                mean_target_accuracy=0.5,
                minimum_target_accuracy=0.5,
                condition_target_accuracies=(0.5,) * 6,
            ).validate()
        with self.assertRaises(ValueError):
            P.Gate8TrainingProbeMetrics(
                probe="head_only",
                step=512,
                message_accuracy=0.9,
                answer_accuracy=0.9,
                activity_accuracy=0.9,
                mean_target_accuracy=0.9,
                minimum_target_accuracy=0.9,
                message_root_invariance=0.9,
                answer_root_invariance=0.9,
            ).validate()


if __name__ == "__main__":
    unittest.main()
