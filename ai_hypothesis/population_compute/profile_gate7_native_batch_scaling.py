"""Engineering-only physical-batch sweep for the Gate-7 native routing bank.

No scientific world, checkpoint, hidden answer, training, compiler, CUDA graph, or mixed precision is used.
The logical routing treatment is held fixed while physical batch grows, measuring whether independent
worlds can amortize the small-kernel routing path on CUDA.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from .profile_gate7_native_bank_scaling import _measure_condition

PROFILE_VERSION = "gate7-native-physical-batch-profile-v0"
POPULATION = 131_072
SLOTS = 128
REPEATS = 5
BATCH_LADDER = (8, 16, 32, 64, 128, 256)
CONDITIONS = (
    ("bounded_score", 16),
    ("bounded_hash", 16),
    ("bounded_score", 64),
    ("bounded_score", 256),
    ("global_score", None),
)


def _condition_key(mode: str, k: int | None) -> str:
    return mode if k is None else f"{mode}_k{k}"


def _recover_cuda_after_oom() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _run_condition(*, batch_size: int, mode: str, k: int | None, device: torch.device) -> dict[str, object]:
    try:
        row = _measure_condition(
            population=POPULATION,
            batch_size=batch_size,
            slots=SLOTS,
            mode=mode,
            k=k,
            repeats=REPEATS,
            device=device,
        )
        return {"status": "OK", **row}
    except torch.OutOfMemoryError as exc:
        _recover_cuda_after_oom()
        return {
            "status": "CUDA_OOM",
            "population": POPULATION,
            "batch_size": batch_size,
            "slots": SLOTS,
            "mode": mode,
            "k": k,
            "error": str(exc),
        }
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        _recover_cuda_after_oom()
        return {
            "status": "CUDA_OOM",
            "population": POPULATION,
            "batch_size": batch_size,
            "slots": SLOTS,
            "mode": mode,
            "k": k,
            "error": str(exc),
        }


def _summaries(rows: list[dict[str, object]]) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for mode, k in CONDITIONS:
        key = _condition_key(mode, k)
        valid = [
            row
            for row in rows
            if row["status"] == "OK" and row["mode"] == mode and row["k"] == k
        ]
        if not valid:
            continue
        valid.sort(key=lambda row: int(row["batch_size"]))
        first = valid[0]
        last = valid[-1]
        first_throughput = float(first["mean_world_decisions_per_second"])
        last_throughput = float(last["mean_world_decisions_per_second"])
        first_latency = float(first["mean_microseconds_per_world_decision"])
        last_latency = float(last["mean_microseconds_per_world_decision"])
        result[key] = {
            "first_batch": int(first["batch_size"]),
            "last_successful_batch": int(last["batch_size"]),
            "throughput_gain": last_throughput / first_throughput,
            "latency_ratio": last_latency / first_latency,
            "first_world_decisions_per_second": first_throughput,
            "last_world_decisions_per_second": last_throughput,
            "first_us_per_world_decision": first_latency,
            "last_us_per_world_decision": last_latency,
            "last_peak_allocated_bytes": int(last["peak_allocated_bytes_max"]),
        }
    return result


def _print_compact(payload: dict[str, object], *, output: Path) -> None:
    rows = payload["rows"]
    assert isinstance(rows, list)
    print("Gate-7 native physical-batch profile — ENGINEERING ONLY")
    print(
        f"GPU: {payload['cuda_device_name']} | N={POPULATION} | slots={SLOTS} | repeats={REPEATS}"
    )
    print("latency = mean microseconds/world decision; throughput = world decisions/s")
    for batch in BATCH_LADDER:
        fields = [f"B={batch:3d}"]
        for mode, k in CONDITIONS:
            row = next(
                row
                for row in rows
                if row["batch_size"] == batch and row["mode"] == mode and row["k"] == k
            )
            label = "global" if k is None else (f"hashK{k}" if mode == "bounded_hash" else f"K{k}")
            if row["status"] != "OK":
                fields.append(f"{label}=OOM")
                continue
            fields.append(
                f"{label}={float(row['mean_microseconds_per_world_decision']):7.3f}us/"
                f"{float(row['mean_world_decisions_per_second']):9.0f}s^-1"
            )
        print("  ".join(fields))

    print("\nBatch-8 -> largest-successful-batch scaling:")
    summaries = payload["summaries"]
    assert isinstance(summaries, dict)
    for key, row in summaries.items():
        assert isinstance(row, dict)
        print(
            f"  {key:22s} B{int(row['first_batch'])}->{int(row['last_successful_batch'])}: "
            f"throughput {float(row['throughput_gain']):.3f}x, "
            f"latency {float(row['latency_ratio']):.3f}x, "
            f"peak {int(row['last_peak_allocated_bytes']) / (1024 ** 3):.2f} GiB"
        )
    print(f"\nFull JSON: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Engineering-only Gate-7 physical-batch scaling profile")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the Gate-7 physical-batch profile")

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    rows: list[dict[str, object]] = []
    for batch_size in BATCH_LADDER:
        for mode, k in CONDITIONS:
            rows.append(
                _run_condition(
                    batch_size=batch_size,
                    mode=mode,
                    k=k,
                    device=device,
                )
            )

    payload: dict[str, object] = {
        "profile_version": PROFILE_VERSION,
        "status": "ENGINEERING_ONLY_NOT_SCIENTIFIC_EVIDENCE",
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "population": POPULATION,
        "slots": SLOTS,
        "repeats": REPEATS,
        "batch_ladder": list(BATCH_LADDER),
        "compiler_enabled": False,
        "cuda_graphs_enabled": False,
        "mixed_precision_enabled": False,
        "scientific_worlds_used": False,
        "checkpoint_used": False,
        "rows": rows,
    }
    payload["summaries"] = _summaries(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _print_compact(payload, output=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
