from __future__ import annotations

import copy
import unittest

from ai_hypothesis.population_language import post_training_learning_l0_protocol as p
from ai_hypothesis.population_language import post_training_learning_l0_world as w


def metric(direct: float, composition: float, retention: float):
    return {
        "direct_episodes": w.DIRECT_HOLDOUT_EXAMPLES,
        "composition_episodes": w.TEST_EXAMPLES,
        "retention_episodes": w.RETENTION_TEST_EPISODES,
        "direct_accuracy": direct,
        "composition_accuracy": composition,
        "retention_accuracy": retention,
    }


def rows(gain: float = 0.03, retention_drop: float = 0.002):
    result = []
    calibration = w.calibration_world_fingerprints()
    for model_seed, world_seed in zip(w.MODEL_INITIALIZATION_SEEDS, w.FINAL_WORLD_SEEDS):
        before = metric(0.20, 0.18, 0.90)
        after = metric(0.24, 0.18 + gain, 0.90 - retention_drop)
        result.append({
            "model_seed": model_seed,
            "world_seed": world_seed,
            "source_reference_implementation_head": p.SOURCE_REFERENCE_IMPLEMENTATION_HEAD,
            "base_model": p.BASE_MODEL,
            "base_parameter_count": p.BASE_PARAMETER_COUNT,
            "base_checkpoint_sha256": "a" * 64,
            "base_checkpoint_after_sha256": "a" * 64,
            "adaptation_artifact_sha256": "b" * 64,
            "adapter_initialization_seed": p.adapter_initialization_seed(model_seed),
            "adaptation_examples": w.ADAPTATION_EXAMPLES,
            "trainable_adaptation_parameters": 180_176,
            "persisted_adaptation_bytes": 720_704,
            "adaptation_updates": 64,
            "adaptation_example_presentations": 512,
            "calibration_world_seeds": list(w.CALIBRATION_WORLD_SEEDS),
            "calibration_world_fingerprints": copy.deepcopy(calibration),
            "final_world_labels_used_during_calibration": False,
            "direct_holdout_used_for_selection": False,
            "validation_used_for_selection": False,
            "test_used_for_selection": False,
            "raw_adaptation_examples_available_at_evaluation": False,
            "external_retrieval_enabled_at_evaluation": False,
            "transient_state_cleared_before_restart": True,
            "base_checkpoint_loaded_fresh_after_restart": True,
            "adaptation_artifact_loaded_after_restart": True,
            "fresh_process_restart": True,
            "adaptation_artifact_payload_kind": p.ARTIFACT_PAYLOAD_KIND,
            "adaptation_artifact_contains_raw_examples": False,
            "adaptation_uses_only_declared_examples_and_gradients": True,
            "world_seed_available_to_adaptation": False,
            "world_rule_parameters_available_to_adaptation": False,
            "world_generator_imported_by_model_runtime": False,
            "symbolic_rule_fitting_used": False,
            "symbolic_execution_used_at_evaluation": False,
            "model_logits_are_authoritative": True,
            "paired_procedure": p.PAIRED_PROCEDURE,
            "paired_rng": p.PAIRED_RNG,
            "paired_quantile_method": p.PAIRED_QUANTILE_METHOD,
            "paired_bootstrap_resamples": p.PAIRED_BOOTSTRAP_RESAMPLES,
            "paired_bootstrap_lower_percentile": p.PAIRED_BOOTSTRAP_LOWER_PERCENTILE,
            "paired_bootstrap_seed": p.paired_bootstrap_seed(world_seed),
            "paired_composition_gain_ci95_lower": gain / 2,
            "final_world_fingerprints": w.final_world_fingerprints(world_seed),
            "baseline": before,
            "post_adaptation_immediate": copy.deepcopy(after),
            "post_restart": copy.deepcopy(after),
        })
    return result


class Contracts(unittest.TestCase):
    def test_worlds_and_protocol_validate(self):
        self.assertTrue(p.validate_protocol()["valid"])
        for seed in (*w.CALIBRATION_WORLD_SEEDS, *w.FINAL_WORLD_SEEDS):
            world = w.make_world(seed)
            self.assertEqual(len({(r.multiplier, r.offset) for r in world.operators}), 8)
            self.assertTrue(all(len({r.apply(x) for x in range(16)}) == 16 for r in world.operators))

    def test_direct_splits_are_disjoint(self):
        for seed in (*w.CALIBRATION_WORLD_SEEDS, *w.FINAL_WORLD_SEEDS):
            a = {(w.make_example("adaptation", i, seed).operators, w.make_example("adaptation", i, seed).input_value) for i in range(64)}
            h = {(w.make_example("direct_holdout", i, seed).operators, w.make_example("direct_holdout", i, seed).input_value) for i in range(64)}
            self.assertTrue(a.isdisjoint(h))
            self.assertEqual(len(a | h), 128)

    def test_world_roles_and_depths(self):
        with self.assertRaises(ValueError):
            w.make_example("calibration", 0, w.FINAL_WORLD_SEEDS[0])
        with self.assertRaises(ValueError):
            w.make_example("test", 0, w.CALIBRATION_WORLD_SEEDS[0])
        for split, depth, seed in (("calibration", 2, 210100), ("validation", 3, 220100), ("test", 4, 220100)):
            self.assertEqual(len(w.make_example(split, 0, seed).operators), depth)

    def test_fingerprints_are_deterministic(self):
        first = w.split_fingerprint("test", 256, 220100)
        self.assertEqual(first, w.split_fingerprint("test", 256, 220100))
        self.assertNotEqual(first, w.split_fingerprint("test", 256, 220101))

    def test_valid_positive_and_negative_results(self):
        positive = rows()
        self.assertEqual(p.classify_run(positive), p.RUN_VALID)
        self.assertEqual(p.learning_summary(positive)["conclusion"], p.SUPPORTS)
        negative = rows(gain=0.005)
        for row in negative:
            row["paired_composition_gain_ci95_lower"] = -0.001
        self.assertEqual(p.classify_run(negative), p.RUN_VALID)
        self.assertEqual(p.learning_summary(negative)["conclusion"], p.DOES_NOT_SUPPORT)

    def test_invalid_boundary_mutations_fail_closed(self):
        mutations = (
            ("base_checkpoint_after_sha256", "c" * 64),
            ("world_seed_available_to_adaptation", True),
            ("symbolic_rule_fitting_used", True),
            ("external_retrieval_enabled_at_evaluation", True),
            ("paired_quantile_method", "nearest"),
        )
        for field, value in mutations:
            candidate = rows()
            candidate[0][field] = value
            self.assertEqual(p.classify_run(candidate), p.RUN_INVALID, field)

    def test_restart_and_fingerprint_tampering_fail_closed(self):
        candidate = rows()
        candidate[0]["post_restart"]["composition_accuracy"] += 0.002
        self.assertEqual(p.classify_run(candidate), p.RUN_INVALID)
        candidate = rows()
        candidate[0]["final_world_fingerprints"]["test"] = "0" * 64
        self.assertEqual(p.classify_run(candidate), p.RUN_INVALID)
        candidate = rows()
        candidate[0]["calibration_world_fingerprints"]["210100"]["adaptation"] = "0" * 64
        self.assertEqual(p.classify_run(candidate), p.RUN_INVALID)

    def test_seed_pairing_and_retention_threshold_are_exact(self):
        reversed_rows = rows()
        reversed_rows.reverse()
        self.assertEqual(p.classify_run(reversed_rows), p.RUN_INVALID)
        self.assertEqual(p.learning_summary(rows(retention_drop=0.005))["conclusion"], p.SUPPORTS)
        self.assertEqual(p.learning_summary(rows(retention_drop=0.006))["conclusion"], p.DOES_NOT_SUPPORT)


if __name__ == "__main__":
    unittest.main()
