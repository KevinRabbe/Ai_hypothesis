from __future__ import annotations

import unittest

import torch

from ai_hypothesis.population_compute.gate3_v1_model import Gate3V1NeuralCandidate, Gate3V1Scorer
from ai_hypothesis.population_compute.gate6_fixed_k_population_scaling import (
    GATE6_BOOTSTRAP_SAMPLES,
    GATE6_CONDITIONS,
    GATE6_DEPTH,
    GATE6_DESCRIPTIVE_K,
    GATE6_EVAL_BATCH_SIZE,
    GATE6_FULL_FRONTIER,
    GATE6_NONINFERIORITY_MARGIN,
    GATE6_POPULATION_LADDER,
    GATE6_PRIMARY_K,
    GATE6_STAGE_A_PARENT_SLOTS,
    GATE6_STAGE_B_PARENT_SLOTS,
    GATE6_TOTAL_LEARNED_UPDATES,
    GATE6_WORLD_COUNT,
    Gate6SchedulerMode,
    _initial_thinning,
    _prune_to_capacity,
    classify_gate6,
    generate_gate6_development_world,
    run_gate6_world_batch,
)


def _candidate(path: tuple[int, ...], score: float) -> Gate3V1NeuralCandidate:
    return Gate3V1NeuralCandidate(path=path, state=torch.zeros(64), score=score)


def _frontier() -> tuple[Gate3V1NeuralCandidate, ...]:
    rows = []
    for value in range(256):
        path = tuple((value >> shift) & 1 for shift in reversed(range(8)))
        rows.append(_candidate(path, float(value)))
    return tuple(rows)


def _metric(checkpoint: int, population: int, comparison: str) -> str:
    return f"c{checkpoint}_n{population}_{comparison}"


def _base_classifier_maps() -> tuple[dict[str, float], dict[str, float]]:
    lows: dict[str, float] = {}
    highs: dict[str, float] = {}
    for checkpoint in (0, 1, 2):
        for population in (64, 128, 256):
            learned = _metric(checkpoint, population, "bounded_score_k16_vs_bounded_hash_k16")
            global_gap = _metric(checkpoint, population, "bounded_score_k16_vs_global_score")
            descriptive = _metric(checkpoint, population, "bounded_score_k8_vs_global_score")
            lows[learned] = 0.10
            highs[learned] = 0.20
            lows[global_gap] = -0.02
            highs[global_gap] = 0.02
            lows[descriptive] = -0.10
            highs[descriptive] = 0.02
    return lows, highs


class Gate6ProtocolTests(unittest.TestCase):
    def test_frozen_constants(self) -> None:
        self.assertEqual(GATE6_DEPTH, 10)
        self.assertEqual(GATE6_FULL_FRONTIER, 256)
        self.assertEqual(GATE6_POPULATION_LADDER, (64, 128, 256))
        self.assertEqual(GATE6_WORLD_COUNT, 256)
        self.assertEqual(GATE6_EVAL_BATCH_SIZE, 64)
        self.assertEqual(GATE6_BOOTSTRAP_SAMPLES, 2000)
        self.assertEqual(GATE6_STAGE_A_PARENT_SLOTS, 255)
        self.assertEqual(GATE6_STAGE_B_PARENT_SLOTS, 128)
        self.assertEqual(GATE6_TOTAL_LEARNED_UPDATES, 6128)
        self.assertEqual(GATE6_PRIMARY_K, 16)
        self.assertEqual(GATE6_DESCRIPTIVE_K, 8)
        self.assertEqual(GATE6_NONINFERIORITY_MARGIN, 0.05)
        self.assertEqual(len(GATE6_CONDITIONS), 4)

    def test_world_namespace_is_deterministic(self) -> None:
        first = generate_gate6_development_world(world_index=0)
        again = generate_gate6_development_world(world_index=0)
        other = generate_gate6_development_world(world_index=1)
        self.assertEqual(first, again)
        self.assertNotEqual(first.public.seed, other.public.seed)
        self.assertEqual(first.public.depth, 10)
        self.assertEqual(len(first.hidden_path), 10)

    def test_initial_thinning_is_nested_and_score_blind(self) -> None:
        frontier = _frontier()
        reversed_scores = tuple(
            Gate3V1NeuralCandidate(path=row.path, state=row.state, score=-row.score)
            for row in frontier
        )
        selected = {}
        selected_reversed = {}
        for population in GATE6_POPULATION_LADDER:
            selected[population] = {
                row.path for row in _initial_thinning(frontier, world_seed=12345, population_size=population)
            }
            selected_reversed[population] = {
                row.path
                for row in _initial_thinning(
                    reversed_scores,
                    world_seed=12345,
                    population_size=population,
                )
            }
            self.assertEqual(selected[population], selected_reversed[population])
        self.assertLess(selected[64], selected[128])
        self.assertLess(selected[128], selected[256])

    def test_capacity_pruning_is_score_blind(self) -> None:
        frontier = _frontier()
        retained, pruned = _prune_to_capacity(
            frontier,
            world_seed=7,
            slot_index=3,
            population_size=64,
        )
        changed_scores = tuple(
            Gate3V1NeuralCandidate(path=row.path, state=row.state, score=1000.0 - row.score)
            for row in frontier
        )
        retained_changed, pruned_changed = _prune_to_capacity(
            changed_scores,
            world_seed=7,
            slot_index=3,
            population_size=64,
        )
        self.assertEqual(pruned, 192)
        self.assertEqual(pruned_changed, 192)
        self.assertEqual({row.path for row in retained}, {row.path for row in retained_changed})

    def test_actual_runtime_enforces_n64_and_k16(self) -> None:
        torch.manual_seed(0)
        model = Gate3V1Scorer()
        world = generate_gate6_development_world(world_index=0)
        result = run_gate6_world_batch(
            model,
            (world,),
            population_size=64,
            mode=Gate6SchedulerMode.BOUNDED_SCORE_K16,
            device="cpu",
        )[0]
        telemetry = result.telemetry
        self.assertEqual(telemetry.stage_a_parent_slots, 255)
        self.assertEqual(telemetry.stage_b_productive_slots, 128)
        self.assertEqual(telemetry.total_learned_updates, 6128)
        self.assertEqual(telemetry.stage_a_frontier_width, 256)
        self.assertEqual(telemetry.initial_stage_b_population_size, 64)
        self.assertEqual(telemetry.population_capacity, 64)
        self.assertEqual(len(telemetry.stage_b_live_population_by_slot), 128)
        self.assertTrue(all(1 <= value <= 64 for value in telemetry.stage_b_live_population_by_slot))
        self.assertTrue(
            all(
                visible == min(16, live)
                for visible, live in zip(
                    telemetry.stage_b_visible_candidate_count_by_slot,
                    telemetry.stage_b_live_population_by_slot,
                    strict=True,
                )
            )
        )
        self.assertEqual(telemetry.stage_b_score_observation_count_by_slot, telemetry.stage_b_visible_candidate_count_by_slot)

    def test_hash_control_observes_zero_scores(self) -> None:
        torch.manual_seed(1)
        model = Gate3V1Scorer()
        world = generate_gate6_development_world(world_index=1)
        result = run_gate6_world_batch(
            model,
            (world,),
            population_size=64,
            mode=Gate6SchedulerMode.BOUNDED_HASH_K16,
            device="cpu",
        )[0]
        self.assertTrue(all(value == 0 for value in result.telemetry.stage_b_score_observation_count_by_slot))


class Gate6ClassifierTests(unittest.TestCase):
    def test_s2_clean_scaling(self) -> None:
        lows, highs = _base_classifier_maps()
        self.assertEqual(classify_gate6(lows, highs), "G6_S2_ROBUST_FIXED_K_POPULATION_SCALING")

    def test_s4_harmful_precedence(self) -> None:
        lows, highs = _base_classifier_maps()
        key = _metric(0, 256, "bounded_score_k16_vs_bounded_hash_k16")
        lows[key] = -0.2
        highs[key] = -0.01
        self.assertEqual(classify_gate6(lows, highs), "G6_S4_BOUNDED_ROUTING_HARMFUL_AT_SCALE")

    def test_s3_checkpoint_sensitive(self) -> None:
        lows, highs = _base_classifier_maps()
        key = _metric(1, 128, "bounded_score_k16_vs_global_score")
        lows[key] = -0.08
        highs[key] = -0.01
        self.assertEqual(classify_gate6(lows, highs), "G6_S3_CHECKPOINT_SENSITIVE_SCALING")

    def test_s0_base_not_established(self) -> None:
        lows, highs = _base_classifier_maps()
        for checkpoint in (0, 1, 2):
            key = _metric(checkpoint, 64, "bounded_score_k16_vs_global_score")
            lows[key] = -0.08
            highs[key] = -0.01
        self.assertEqual(classify_gate6(lows, highs), "G6_S0_FIXED_K_NOT_ESTABLISHED")

    def test_s1_uniform_degradation(self) -> None:
        lows, highs = _base_classifier_maps()
        for checkpoint in (0, 1, 2):
            key = _metric(checkpoint, 256, "bounded_score_k16_vs_global_score")
            lows[key] = -0.08
            highs[key] = -0.01
        self.assertEqual(classify_gate6(lows, highs), "G6_S1_FIXED_K_DEGRADES_WITH_POPULATION")


if __name__ == "__main__":
    unittest.main()
