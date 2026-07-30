"""Admitted execution surface for Gate-7 information-ceiling precision confirmation."""

from .gate7_information_ceiling_precision_confirmation_protocol import *  # noqa: F403
from .gate7_information_ceiling_precision_confirmation_rank import (
    Gate7PrecisionBatchRanks,
    Gate7PrecisionCheckpointRanks,
    aggregate_gate7_precision_rank_batches,
    evaluate_gate7_precision_rank_batch,
)
from .gate7_information_ceiling_precision_confirmation_statistics import (
    Gate7PrecisionBootstrapSummary,
    Gate7PrecisionRankSummary,
    classify_gate7_precision_from_rank_matrix,
    gate7_precision_cell_statistics,
    gate7_precision_pooled_statistics,
    gate7_precision_population_statistics,
    summarize_gate7_precision_ranks,
)
from .gate7_information_ceiling_precision_confirmation_worlds import (
    Gate7PrecisionWorld,
    generate_gate7_precision_world,
    load_verified_gate7_precision_checkpoint,
    precision_world_batch,
)

GATE7_PRECISION_PROTOCOL_HEAD = "8d7865ab01b4b04b875ed2ca627b68a6c33c81f7"
GATE7_PRECISION_EXECUTION_ADMITTED = True
GATE7_PRECISION_EXECUTION_SCIENTIFIC_STATUS = (
    "FRESH_GATE7_INFORMATION_CEILING_PRECISION_CONFIRMATION_EVIDENCE"
)

__all__ = [name for name in globals() if name.startswith("GATE7_")] + [
    "Gate7PrecisionWorld",
    "Gate7PrecisionBatchRanks",
    "Gate7PrecisionCheckpointRanks",
    "Gate7PrecisionRankSummary",
    "Gate7PrecisionBootstrapSummary",
    "generate_gate7_precision_world",
    "precision_world_batch",
    "load_verified_gate7_precision_checkpoint",
    "evaluate_gate7_precision_rank_batch",
    "aggregate_gate7_precision_rank_batches",
    "summarize_gate7_precision_ranks",
    "gate7_precision_cell_statistics",
    "gate7_precision_population_statistics",
    "gate7_precision_pooled_statistics",
    "classify_gate7_precision_from_rank_matrix",
]
