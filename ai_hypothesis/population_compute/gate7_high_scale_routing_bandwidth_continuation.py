"""Admitted execution surface for the frozen Gate-7 post-confirmation continuation."""

from .gate7_high_scale_routing_bandwidth_continuation_conditions import (
    Gate7ContinuationBatchCondition,
    Gate7ContinuationCondition,
    aggregate_gate7_continuation_condition,
    build_gate7_continuation_frontier,
    evaluate_gate7_continuation_batch_condition,
)
from .gate7_high_scale_routing_bandwidth_continuation_statistics import (
    GATE7_CONTINUATION_PROTOCOL_HEAD,
    Gate7ContinuationPairedSummary,
    Gate7ContinuationStratifiedSummary,
    continuation_provenance,
    paired_gate7_continuation_summary,
    stratified_gate7_continuation_global_summary,
)
from .gate7_high_scale_routing_bandwidth_continuation_worlds import (
    Gate7ContinuationWorld,
    continuation_world_batch,
    gate7_continuation_runtime_seed,
    generate_gate7_continuation_world,
    load_verified_gate7_continuation_checkpoint,
)

GATE7_CONTINUATION_EXECUTION_ADMITTED = True
GATE7_CONTINUATION_SCIENTIFIC_STATUS = (
    "FRESH_HIGH_SCALE_ROUTING_BANDWIDTH_CONTINUATION_EVIDENCE"
)

__all__ = [
    "GATE7_CONTINUATION_EXECUTION_ADMITTED",
    "GATE7_CONTINUATION_SCIENTIFIC_STATUS",
    "GATE7_CONTINUATION_PROTOCOL_HEAD",
    "Gate7ContinuationWorld",
    "Gate7ContinuationBatchCondition",
    "Gate7ContinuationCondition",
    "Gate7ContinuationPairedSummary",
    "Gate7ContinuationStratifiedSummary",
    "gate7_continuation_runtime_seed",
    "generate_gate7_continuation_world",
    "continuation_world_batch",
    "load_verified_gate7_continuation_checkpoint",
    "build_gate7_continuation_frontier",
    "evaluate_gate7_continuation_batch_condition",
    "aggregate_gate7_continuation_condition",
    "paired_gate7_continuation_summary",
    "stratified_gate7_continuation_global_summary",
    "continuation_provenance",
]
