from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai_hypothesis.population_compute.audit_relay_resource_frontier import main as audit_main
from ai_hypothesis.population_compute.relay_resource_audit import (
    CANONICAL_BATCH_SIZES,
    CANONICAL_DIFFICULTIES,
    CANONICAL_POPULATION_SIZES,
    audit_relay_resource_result,
    render_relay_resource_markdown,
)
from ai_hypothesis.population_compute.relay_resource_frontier import (
    RESOURCE_FRONTIER_VERSION,
    _measurement_rotation,
)


SCHEDULES = (
    "parallel_normalized",
    "serial_normalized",
    "serial_cached_normalized",
)
FINGERPRINT = "a" * 64


def schedule(
    name: str,
    *,
    active_workers: int,
    relay_hops: int,
    batch_size: int,
    device_type: str = "cuda",
) -> dict[str, object]:
    parallel_or_cached = name != "serial_normalized"
    static = 2 * active_workers if parallel_or_cached else 2 * active_workers * relay_hops
    cached_vectors = active_workers if name != "serial_normalized" else 0
    return {
        "schedule": name,
        "batch_size": batch_size,
        "warmup_iterations": 20,
        "measured_iterations": 100,
        "median_batch_latency_ms": 1.0,
        "p95_batch_latency_ms": 1.2,
        "min_batch_latency_ms": 0.9,
        "total_measured_seconds": 0.1,
        "throughput_samples_per_second": 1000.0,
        "latency_source": "cuda_event" if device_type == "cuda" else "host_perf_counter",
        "host_enqueue_seconds": 0.05,
        "device_median_latency_ms": 0.95 if device_type == "cuda" else None,
        "cuda_baseline_allocated_bytes": 1000 if device_type == "cuda" else None,
        "cuda_peak_allocated_bytes": 2000 if device_type == "cuda" else None,
        "cuda_peak_allocated_delta_bytes": 1000 if device_type == "cuda" else None,
        "cuda_baseline_reserved_bytes": 3000 if device_type == "cuda" else None,
        "cuda_peak_reserved_bytes": 4000 if device_type == "cuda" else None,
        "worker_updates_per_sample": active_workers * relay_hops,
        "candidate_evaluations_per_sample": active_workers * relay_hops,
        "input_projection_evaluations_per_sample": (
            active_workers if parallel_or_cached else active_workers * relay_hops
        ),
        "value_projection_evaluations_per_sample": (
            active_workers if parallel_or_cached else active_workers * relay_hops
        ),
        "static_projection_evaluations_per_sample": static,
        "communicated_scalars_per_sample": 16 if name == "parallel_normalized" else 0,
        "peak_active_neural_states_per_sample": (
            active_workers if name == "parallel_normalized" else 1
        ),
        "cached_state_vectors_per_sample": cached_vectors,
        "cached_message_vectors_per_sample": cached_vectors,
    }


def canonical_payload(*, device_type: str = "cuda") -> dict[str, object]:
    comparisons: list[dict[str, object]] = []
    for difficulty, relay_hops in CANONICAL_DIFFICULTIES.items():
        for batch_size in CANONICAL_BATCH_SIZES:
            for active_workers in CANONICAL_POPULATION_SIZES:
                rotation = _measurement_rotation(
                    difficulty=difficulty,
                    active_workers=active_workers,
                    batch_size=batch_size,
                    relay_hops=relay_hops,
                )
                order = SCHEDULES[rotation:] + SCHEDULES[:rotation]
                comparisons.append(
                    {
                        "difficulty": difficulty,
                        "relay_hops": relay_hops,
                        "active_workers": active_workers,
                        "batch_size": batch_size,
                        "learned_parameter_count": 26_669,
                        "parameter_fingerprint": FINGERPRINT,
                        "outputs_equivalent": True,
                        "decoded_predictions_equal": True,
                        "max_abs_logits_difference": 1e-6,
                        "max_abs_shared_difference": 1e-6,
                        "recurrent_worker_updates_equal": True,
                        "parallel_cached_static_projection_work_equal": True,
                        "parallel_learned_span_proxy": relay_hops,
                        "serial_low_memory_learned_span_proxy": active_workers * relay_hops,
                        "serial_cached_learned_span_proxy": active_workers * relay_hops,
                        "measurement_order": list(order),
                        "parallel": schedule(
                            "parallel_normalized",
                            active_workers=active_workers,
                            relay_hops=relay_hops,
                            batch_size=batch_size,
                            device_type=device_type,
                        ),
                        "serial_low_memory": schedule(
                            "serial_normalized",
                            active_workers=active_workers,
                            relay_hops=relay_hops,
                            batch_size=batch_size,
                            device_type=device_type,
                        ),
                        "serial_cached": schedule(
                            "serial_cached_normalized",
                            active_workers=active_workers,
                            relay_hops=relay_hops,
                            batch_size=batch_size,
                            device_type=device_type,
                        ),
                        "low_memory_serial_over_parallel_latency_speedup": 1.50,
                        "cached_serial_over_parallel_latency_speedup": 1.25,
                    }
                )
    return {
        "benchmark_version": RESOURCE_FRONTIER_VERSION,
        "checkpoint": {
            "experiment_version": "population-compute-relay-training-v1",
            "protocol_version": "relay-protocol-v1-normalized-gate-supervised",
            "benchmark_version": "collective-relay-v1-answer-frontier",
            "training_seed": 1,
            "parameter_fingerprint": FINGERPRINT,
        },
        "config": {
            "population_sizes": list(CANONICAL_POPULATION_SIZES),
            "batch_sizes": list(CANONICAL_BATCH_SIZES),
            "warmup_iterations": 20,
            "measured_iterations": 100,
            "world_seed": 0,
            "equivalence_rtol": 2e-5,
            "equivalence_atol": 2e-5,
        },
        "device": device_type,
        "learned_parameter_count": 26_669,
        "parameter_fingerprint": FINGERPRINT,
        "provenance": {
            "device_type": device_type,
            "execution_mode": "eager",
            "schedule_timing_policy": (
                "each schedule independently warmed; measurement order rotates deterministically by condition"
            ),
        },
        "comparisons": comparisons,
    }


class RelayResourceAuditTests(unittest.TestCase):
    def test_complete_canonical_cuda_matrix_is_protocol_valid_without_gate_threshold(self) -> None:
        payload = canonical_payload()
        audit = audit_relay_resource_result(payload)

        self.assertTrue(audit.protocol_valid, audit.reasons)
        self.assertEqual(audit.observed_condition_count, 30)
        self.assertEqual(audit.observed_measurement_rotations, (0, 1, 2))
        self.assertEqual(len(audit.batch_summaries), 2)
        self.assertTrue(
            all(summary.parallel_faster_than_cached_count == 15 for summary in audit.batch_summaries)
        )
        report = render_relay_resource_markdown(payload, audit)
        self.assertIn("Protocol valid: **YES**", report)
        self.assertIn("No threshold here declares Gate 1 passed", report)
        self.assertEqual(report.count("| 1 | relay-"), 15)
        self.assertEqual(report.count("| 64 | relay-"), 15)

    def test_missing_condition_is_rejected_instead_of_summarized_as_complete(self) -> None:
        payload = canonical_payload()
        payload["comparisons"] = payload["comparisons"][:-1]
        audit = audit_relay_resource_result(payload)

        self.assertFalse(audit.protocol_valid)
        self.assertTrue(any("expected 30 frozen conditions" in reason for reason in audit.reasons))
        self.assertTrue(any("missing frozen conditions" in reason for reason in audit.reasons))

    def test_cpu_result_requires_explicit_non_decisive_override(self) -> None:
        payload = canonical_payload(device_type="cpu")
        decisive = audit_relay_resource_result(payload)
        mechanics = audit_relay_resource_result(payload, require_cuda=False)

        self.assertFalse(decisive.protocol_valid)
        self.assertTrue(any("requires CUDA" in reason for reason in decisive.reasons))
        self.assertTrue(mechanics.protocol_valid, mechanics.reasons)

    def test_static_projection_or_rotation_contract_break_is_rejected(self) -> None:
        payload = canonical_payload()
        first = payload["comparisons"][0]
        first["serial_cached"]["static_projection_evaluations_per_sample"] += 2
        first["parallel_cached_static_projection_work_equal"] = False
        for row in payload["comparisons"]:
            row["measurement_order"] = list(SCHEDULES)

        audit = audit_relay_resource_result(payload)
        self.assertFalse(audit.protocol_valid)
        self.assertTrue(any("parallel_cached_static_projection_work_equal" in reason for reason in audit.reasons))
        self.assertTrue(any("all three schedule-order rotations" in reason for reason in audit.reasons))

    def test_frozen_tolerance_row_identity_and_cuda_memory_drift_are_rejected(self) -> None:
        payload = canonical_payload()
        payload["config"]["equivalence_rtol"] = 1e-4
        first = payload["comparisons"][0]
        first["parameter_fingerprint"] = "b" * 64
        first["parallel_learned_span_proxy"] += 1
        first["parallel"]["warmup_iterations"] = 19
        first["parallel"]["cuda_peak_allocated_delta_bytes"] = 999

        audit = audit_relay_resource_result(payload)
        self.assertFalse(audit.protocol_valid)
        self.assertTrue(any("equivalence_rtol" in reason for reason in audit.reasons))
        self.assertTrue(any("parameter fingerprint differs" in reason for reason in audit.reasons))
        self.assertTrue(any("parallel learned-span proxy" in reason for reason in audit.reasons))
        self.assertTrue(any("warm-up count" in reason for reason in audit.reasons))
        self.assertTrue(any("allocated delta is inconsistent" in reason for reason in audit.reasons))

    def test_cli_writes_audit_and_report_and_returns_nonzero_for_invalid_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            source = root / "frontier.json"
            audit_path = root / "audit.json"
            report_path = root / "report.md"
            payload = canonical_payload(device_type="cpu")
            source.write_text(json.dumps(payload), encoding="utf-8")

            exit_code = audit_main(
                [
                    "--input",
                    str(source),
                    "--audit-output",
                    str(audit_path),
                    "--report-output",
                    str(report_path),
                    "--allow-non-cuda",
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue(json.loads(audit_path.read_text())["protocol_valid"])
            self.assertIn("All frozen cells", report_path.read_text())

            payload["comparisons"] = payload["comparisons"][:-1]
            source.write_text(json.dumps(payload), encoding="utf-8")
            exit_code = audit_main(
                [
                    "--input",
                    str(source),
                    "--audit-output",
                    str(audit_path),
                    "--report-output",
                    str(report_path),
                    "--allow-non-cuda",
                ]
            )
            self.assertEqual(exit_code, 2)
            self.assertFalse(json.loads(audit_path.read_text())["protocol_valid"])


if __name__ == "__main__":
    unittest.main()
