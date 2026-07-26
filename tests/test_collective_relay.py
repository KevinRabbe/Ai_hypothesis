from __future__ import annotations

import unittest

from ai_hypothesis.population_compute import (
    COLLECTIVE_RELAY_VERSION,
    RELAY_DIFFICULTIES,
    RelayDifficulty,
    generate_relay_dataset,
    generate_relay_world,
    resolve_relay,
)


class CollectiveRelayTests(unittest.TestCase):
    def test_generation_is_deterministic(self) -> None:
        difficulty = RELAY_DIFFICULTIES[1]

        first = generate_relay_world(1234, difficulty)
        second = generate_relay_world(1234, difficulty)

        self.assertEqual(first, second)
        self.assertEqual(first.version, COLLECTIVE_RELAY_VERSION)
        self.assertEqual(resolve_relay(first), first.answer_key)

    def test_chain_is_distributed_among_shuffled_worker_slots(self) -> None:
        difficulty = RelayDifficulty(name="test", world_size=32, hop_count=4)
        world = generate_relay_world(9, difficulty)
        chain_slots = tuple(
            record.worker_slot for record in world.records if record.is_chain_edge
        )

        self.assertEqual(len(chain_slots), difficulty.hop_count)
        self.assertEqual(len(set(chain_slots)), difficulty.hop_count)
        self.assertNotEqual(chain_slots, tuple(range(difficulty.hop_count)))

    def test_record_keys_are_unique_and_cover_exact_world_size(self) -> None:
        difficulty = RELAY_DIFFICULTIES[-1]
        world = generate_relay_world(77, difficulty)

        self.assertEqual(len(world.records), difficulty.world_size)
        self.assertEqual(len({record.key for record in world.records}), difficulty.world_size)
        self.assertEqual(
            {record.worker_slot for record in world.records},
            set(range(difficulty.world_size)),
        )

    def test_dataset_uses_consecutive_frozen_seeds(self) -> None:
        difficulty = RELAY_DIFFICULTIES[0]
        worlds = generate_relay_dataset(
            start_seed=100,
            world_count=4,
            difficulty=difficulty,
        )

        self.assertEqual(tuple(world.seed for world in worlds), (100, 101, 102, 103))

    def test_hop_count_must_require_collective_relay(self) -> None:
        difficulty = RelayDifficulty(name="bad", world_size=16, hop_count=1)

        with self.assertRaisesRegex(ValueError, "at least 2"):
            difficulty.validate()


if __name__ == "__main__":
    unittest.main()
