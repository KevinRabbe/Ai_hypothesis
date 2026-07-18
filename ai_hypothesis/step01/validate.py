"""Validate the Step 1 procedural benchmark before neural training."""

from __future__ import annotations

from collections import Counter, defaultdict

from .baselines import oracle_label, predict_baselines
from .generator import generate_sample
from .schema import DIFFICULTIES, TASKS, VALID_LABELS


def main() -> None:
    per_bucket = 100
    label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    baseline_correct: dict[str, Counter[str]] = defaultdict(Counter)
    baseline_total: dict[str, Counter[str]] = defaultdict(Counter)

    for task in TASKS:
        for difficulty in DIFFICULTIES:
            for seed in range(per_bucket):
                sample = generate_sample(task, difficulty, seed)
                sample.validate()

                replay = generate_sample(task, difficulty, seed)
                if replay != sample:
                    raise AssertionError(
                        f"non-deterministic replay for {task.value}/{difficulty.value}/{seed}"
                    )

                oracle = oracle_label(sample)
                if oracle != sample.label:
                    raise AssertionError(
                        f"oracle mismatch for {task.value}/{difficulty.value}/{seed}: "
                        f"{oracle} != {sample.label}"
                    )

                label_counts[f"{task.value}:{difficulty.value}"][sample.label] += 1

                for name, prediction in predict_baselines(sample).items():
                    if prediction not in VALID_LABELS[task]:
                        raise AssertionError(
                            f"baseline {name} returned invalid label {prediction!r}"
                        )
                    key = f"{task.value}:{difficulty.value}"
                    baseline_total[key][name] += 1
                    if prediction == sample.label:
                        baseline_correct[key][name] += 1

    print("Step 1 benchmark validation passed.")
    print(f"Validated {len(TASKS) * len(DIFFICULTIES) * per_bucket:,} examples.")
    print()
    for key in sorted(label_counts):
        print(f"{key}")
        print(f"  labels: {dict(label_counts[key])}")
        for name, total in baseline_total[key].items():
            correct = baseline_correct[key][name]
            print(f"  {name}: {correct / total:.3f} accuracy")


if __name__ == "__main__":
    main()
