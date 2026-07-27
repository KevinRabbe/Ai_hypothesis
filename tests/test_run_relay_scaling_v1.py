from __future__ import annotations

import unittest
from unittest.mock import patch

from ai_hypothesis.population_compute.relay_experiment_v1 import RelayTrainingConfigV1
from ai_hypothesis.population_compute.run_relay_scaling_v1 import (
    FROZEN_CONFIRMATION_BATCH_SIZE,
    FROZEN_CONFIRMATION_CONFIG,
    FROZEN_CONFIRMATION_WORLD_COUNT,
    _require_frozen_confirmation_configuration,
    main,
)


class RunRelayScalingV1Tests(unittest.TestCase):
    def test_confirmation_is_refused_before_training_without_explicit_unlock(self) -> None:
        with patch(
            "ai_hypothesis.population_compute.run_relay_scaling_v1.train_relay_model_v1"
        ) as train:
            with self.assertRaisesRegex(
                SystemExit,
                "Refusing to open frozen confirmation",
            ):
                main(
                    [
                        "--evaluation-split",
                        "confirmation",
                        "--output-dir",
                        "unused",
                    ]
                )
        train.assert_not_called()

    def test_confirmation_rejects_changed_training_configuration(self) -> None:
        changed = RelayTrainingConfigV1(steps=FROZEN_CONFIRMATION_CONFIG.steps + 1)
        with self.assertRaisesRegex(SystemExit, "frozen canonical relay-v1"):
            _require_frozen_confirmation_configuration(
                changed,
                evaluation_world_count=FROZEN_CONFIRMATION_WORLD_COUNT,
                evaluation_batch_size=FROZEN_CONFIRMATION_BATCH_SIZE,
            )

    def test_confirmation_rejects_changed_world_count(self) -> None:
        with self.assertRaisesRegex(SystemExit, "exactly 1000"):
            _require_frozen_confirmation_configuration(
                FROZEN_CONFIRMATION_CONFIG,
                evaluation_world_count=FROZEN_CONFIRMATION_WORLD_COUNT - 1,
                evaluation_batch_size=FROZEN_CONFIRMATION_BATCH_SIZE,
            )

    def test_confirmation_rejects_changed_evaluation_batch_size(self) -> None:
        with self.assertRaisesRegex(SystemExit, "batch size 64"):
            _require_frozen_confirmation_configuration(
                FROZEN_CONFIRMATION_CONFIG,
                evaluation_world_count=FROZEN_CONFIRMATION_WORLD_COUNT,
                evaluation_batch_size=FROZEN_CONFIRMATION_BATCH_SIZE // 2,
            )

    def test_exact_frozen_confirmation_configuration_is_accepted(self) -> None:
        _require_frozen_confirmation_configuration(
            FROZEN_CONFIRMATION_CONFIG,
            evaluation_world_count=FROZEN_CONFIRMATION_WORLD_COUNT,
            evaluation_batch_size=FROZEN_CONFIRMATION_BATCH_SIZE,
        )


if __name__ == "__main__":
    unittest.main()
