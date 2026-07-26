from __future__ import annotations

import unittest

from ai_hypothesis.population_compute import (
    COLLECTIVE_RELAY_VERSION,
    DEVELOPMENT_POPULATION_SIZES,
    RELAY_DIFFICULTIES,
    RELAY_WORLD_SIZE,
    RelayDifficulty,
    complete_chain_ids_at,
    generate_relay_dataset,
    generate_relay_world,
    information_complete_at,
    relay_population_points,
    relay_scope_thresholds,
    resolve_relay,
)


class CollectiveRelayTests(unittest.TestCase):
    def test_generation_is_deterministic(self) -> None:
        difficulty = RELAY_DIFFICULTIES[1]

        first = generate_relay_world(1234, difficulty)
        second = generate_relay_world(1234, difficulty)

        self.assertEqual(first, second)
        self.assertEqual(first.version, COLLECTIVE_RELAY_VERSION)
        self.assertEqual(first.version, "collective-relay-v1")
        self.assertEqual(resolve_relay(first), first.answer_key)

    def test_frozen_difficulties_change_hops_not_world_scope(self) -> None:
        self.assertEqual(
            {difficulty.world_size for difficulty in RELAY_DIFFICULTIES},
            {RELAY_WORLD_SIZE},
        )
        self.assertEqual(RELAY_WORLD_SIZE, DEVELOPMENT_POPULATION_SIZES[-1])
        self.assertEqual(
            tuple(difficulty.hop_count for difficulty in RELAY_DIFFICULTIES),
            (2, 4, 8),
        )
        self.assertEqual(relay_scope_thresholds(RELAY_DIFFICULTIES[0]), (4, 16, 64, 256))
        self.assertEqual(relay_scope_thresholds(RELAY_DIFFICULTIES[1]), (16, 64, 256))
        self.assertEqual(relay_scope_thresholds(RELAY_DIFFICULTIES[2]), (16, 64, 256))

    def test_scope_thresholds_cycle_exactly_over_admissible_population_points(self) -> None:
        for difficulty in RELAY_DIFFICULTIES:
            thresholds = relay_scope_thresholds(difficulty)
            worlds = tuple(
                generate_relay_world(seed, difficulty)
                for seed in range(len(thresholds) * 2)
            )
            self.assertEqual(
                tuple(world.scope_threshold for world in worlds),
                thresholds * 2,
            )

    def test_target_becomes_complete_at_threshold_and_answer_is_hidden_before_it(self) -> None:
        for difficulty in RELAY_DIFFICULTIES:
            population_points = relay_population_points(difficulty)
            thresholds = relay_scope_thresholds(difficulty)
            for seed, threshold in enumerate(thresholds):
                world = generate_relay_world(seed, difficulty)
                self.assertEqual(world.scope_threshold, threshold)
                previous = max(size for size in population_points if size < threshold)
                self.assertFalse(information_complete_at(world, previous))
                self.assertTrue(information_complete_at(world, threshold))
                self.assertTrue(
                    all(
                        information_complete_at(world, larger)
                        for larger in population_points
                        if larger >= threshold
                    )
                )
                self.assertFalse(
                    any(
                        record.key == world.answer_key or record.value == world.answer_key
                        for record in world.records[:previous]
                    )
                )
                final_edge = tuple(
                    record
                    for record in world.records
                    if record.is_chain_edge and record.value == world.answer_key
                )
                self.assertEqual(len(final_edge), 1)
                self.assertGreaterEqual(final_edge[0].worker_slot, previous)

    def test_first_complete_prefix_contains_only_whole_matched_chains(self) -> None:
        for difficulty in RELAY_DIFFICULTIES:
            for seed in range(12):
                world = generate_relay_world(seed, difficulty)
                complete = complete_chain_ids_at(world, world.scope_threshold)
                self.assertIn(0, complete)
                self.assertGreaterEqual(len(complete), 2)
                self.assertEqual(
                    len(complete) * difficulty.hop_count,
                    world.scope_threshold,
                )
                for chain_id in complete:
                    records = tuple(
                        record
                        for record in world.records[: world.scope_threshold]
                        if record.chain_id == chain_id
                    )
                    self.assertEqual(len(records), difficulty.hop_count)
                    self.assertEqual(
                        {record.is_chain_edge for record in records},
                        {chain_id == 0},
                    )

    def test_target_and_distractor_chains_have_identical_local_shape(self) -> None:
        difficulty = RELAY_DIFFICULTIES[2]
        world = generate_relay_world(7, difficulty)
        chain_count = difficulty.world_size // difficulty.hop_count
        for chain_id in range(chain_count):
            records = tuple(record for record in world.records if record.chain_id == chain_id)
            self.assertEqual(len(records), difficulty.hop_count)
            keys = {record.key for record in records}
            values = {record.value for record in records}
            self.assertEqual(len(keys - values), 1)
            self.assertEqual(len(values - keys), 1)

    def test_smaller_custom_world_retains_nested_population_ladder(self) -> None:
        difficulty = RelayDifficulty(name="test", world_size=32, hop_count=4)
        self.assertEqual(relay_population_points(difficulty), (1, 4, 16, 32))
        self.assertEqual(relay_scope_thresholds(difficulty), (16, 32))

        world = generate_relay_world(0, difficulty)
        self.assertEqual(world.scope_threshold, 16)
        self.assertFalse(information_complete_at(world, 4))
        self.assertTrue(information_complete_at(world, 16))
        self.assertEqual(len(complete_chain_ids_at(world, 16)), 4)

    def test_record_keys_are_unique_and_cover_exact_world_size(self) -> None:
        difficulty = RELAY_DIFFICULTIES[-1]
        world = generate_relay_world(77, difficulty)

        self.assertEqual(len(world.records), difficulty.world_size)
        self.assertEqual(len({record.key for record in world.records}), difficulty.world_size)
        self.assertEqual(
            {record.worker_slot for record in world.records},
            set(range(difficulty.world_size)),
        )
        self.assertEqual(
            {record.chain_id for record in world.records},
            set(range(difficulty.world_size // difficulty.hop_count)),
        )

    def test_dataset_uses_consecutive_frozen_seeds(self) -> None:
        difficulty = RELAY_DIFFICULTIES[0]
        worlds = generate_relay_dataset(
            start_seed=100,
            world_count=4,
            difficulty=difficulty,
        )

        self.assertEqual(tuple(world.seed for world in worlds), (100, 101, 102, 103))

    def test_world_requires_room_for_complete_distractor_chain(self) -> None:
        difficulty = RelayDifficulty(name="bad", world_size=8, hop_count=8)

        with self.assertRaisesRegex(ValueError, "at least two complete chains"):
            difficulty.validate()

    def test_hop_count_must_require_collective_relay(self) -> None:
        difficulty = RelayDifficulty(name="bad", world_size=16, hop_count=1)

        with self.assertRaisesRegex(ValueError, "at least 2"):
            difficulty.validate()


if __name__ == "__main__":
    unittest.main()
