from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "ai_hypothesis"
    / "population_compute"
    / "gate6_population_scaling_prep.py"
)
SPEC = importlib.util.spec_from_file_location("gate6_population_scaling_prep_standalone", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load Gate-6 preparation module")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

GATE6_DESCRIPTIVE_K = MODULE.GATE6_DESCRIPTIVE_K
GATE6_FULL_FRONTIER = MODULE.GATE6_FULL_FRONTIER
GATE6_NONINFERIORITY_MARGIN = MODULE.GATE6_NONINFERIORITY_MARGIN
GATE6_POPULATION_LADDER = MODULE.GATE6_POPULATION_LADDER
GATE6_PREPARATION_ONLY = MODULE.GATE6_PREPARATION_ONLY
GATE6_PRIMARY_K = MODULE.GATE6_PRIMARY_K
GATE6_STAGE_A_LEARNED_UPDATES = MODULE.GATE6_STAGE_A_LEARNED_UPDATES
GATE6_STAGE_A_PARENT_SLOTS = MODULE.GATE6_STAGE_A_PARENT_SLOTS
GATE6_STAGE_B_LEARNED_UPDATES = MODULE.GATE6_STAGE_B_LEARNED_UPDATES
GATE6_STAGE_B_PARENT_SLOTS = MODULE.GATE6_STAGE_B_PARENT_SLOTS
GATE6_TOTAL_LEARNED_UPDATES = MODULE.GATE6_TOTAL_LEARNED_UPDATES
build_gate6_preparation_plan = MODULE.build_gate6_preparation_plan
complete_depth8_frontier_paths = MODULE.complete_depth8_frontier_paths
nested_answer_blind_thinning = MODULE.nested_answer_blind_thinning
validate_nested_thinning = MODULE.validate_nested_thinning


class Gate6PreparationTests(unittest.TestCase):
    def test_preparation_only_and_work_identity(self) -> None:
        self.assertTrue(GATE6_PREPARATION_ONLY)
        self.assertEqual(GATE6_FULL_FRONTIER, 256)
        self.assertEqual(GATE6_POPULATION_LADDER, (64, 128, 256))
        self.assertEqual(GATE6_STAGE_A_PARENT_SLOTS, 255)
        self.assertEqual(GATE6_STAGE_B_PARENT_SLOTS, 128)
        self.assertEqual(GATE6_STAGE_A_LEARNED_UPDATES, 4080)
        self.assertEqual(GATE6_STAGE_B_LEARNED_UPDATES, 2048)
        self.assertEqual(GATE6_TOTAL_LEARNED_UPDATES, 6128)
        self.assertEqual(GATE6_PRIMARY_K, 16)
        self.assertEqual(GATE6_DESCRIPTIVE_K, 8)
        self.assertEqual(GATE6_NONINFERIORITY_MARGIN, 0.05)

        signatures = []
        for population_size in GATE6_POPULATION_LADDER:
            plan = build_gate6_preparation_plan(population_size)
            plan.validate()
            signatures.append(
                (
                    plan.stage_a_parent_slots,
                    plan.stage_b_parent_slots,
                    plan.active_child_lanes,
                    plan.recurrent_updates_per_child,
                    plan.total_learned_updates,
                )
            )
        self.assertEqual(len(set(signatures)), 1)

    def test_complete_frontier_is_structural_only(self) -> None:
        frontier = complete_depth8_frontier_paths()
        self.assertEqual(len(frontier), 256)
        self.assertEqual(len(set(frontier)), 256)
        self.assertTrue(all(len(path) == 8 for path in frontier))
        self.assertTrue(all(bit in (0, 1) for path in frontier for bit in path))

    def test_nested_thinning(self) -> None:
        frontier = complete_depth8_frontier_paths()
        for runtime_seed in (0, 1, 7, 123456789):
            n64 = set(nested_answer_blind_thinning(frontier, runtime_seed=runtime_seed, population_size=64))
            n128 = set(nested_answer_blind_thinning(frontier, runtime_seed=runtime_seed, population_size=128))
            n256 = set(nested_answer_blind_thinning(frontier, runtime_seed=runtime_seed, population_size=256))
            self.assertEqual(len(n64), 64)
            self.assertEqual(len(n128), 128)
            self.assertEqual(len(n256), 256)
            self.assertLess(n64, n128)
            self.assertLess(n128, n256)
            validate_nested_thinning(runtime_seed=runtime_seed)

    def test_thinning_is_deterministic_and_seed_sensitive(self) -> None:
        frontier = complete_depth8_frontier_paths()
        first = nested_answer_blind_thinning(frontier, runtime_seed=11, population_size=64)
        again = nested_answer_blind_thinning(frontier, runtime_seed=11, population_size=64)
        other = nested_answer_blind_thinning(frontier, runtime_seed=12, population_size=64)
        self.assertEqual(first, again)
        self.assertNotEqual(first, other)

    def test_invalid_population_size_is_rejected(self) -> None:
        frontier = complete_depth8_frontier_paths()
        with self.assertRaises(ValueError):
            nested_answer_blind_thinning(frontier, runtime_seed=0, population_size=32)


if __name__ == "__main__":
    unittest.main()
