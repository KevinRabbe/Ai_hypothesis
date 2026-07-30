"""Admitted execution surface for the frozen Gate-7 information-ceiling decomposition."""

from .gate7_information_ceiling_decomposition_protocol import *  # noqa: F403
from .gate7_information_ceiling_decomposition_rank import (
    Gate7InformationCeilingBatchRanks,
    Gate7InformationCeilingCheckpointRanks,
    aggregate_information_ceiling_rank_batches,
    evaluate_information_ceiling_rank_batch,
)
from .gate7_information_ceiling_decomposition_statistics import (
    Gate7InformationCeilingPairedSummary,
    Gate7InformationCeilingRankSummary,
    classify_from_paired_summaries,
    comparison_for_frozen_classifier,
    paired_information_ceiling_summary,
    summarize_information_ceiling_ranks,
)
from .gate7_information_ceiling_decomposition_worlds import (
    Gate7InformationCeilingWorld,
    generate_gate7_information_ceiling_world,
    information_ceiling_world_batch,
    load_verified_gate7_information_ceiling_checkpoint,
)

GATE7_INFORMATION_CEILING_EXECUTION_ADMITTED = True
GATE7_INFORMATION_CEILING_SCIENTIFIC_STATUS = (
    "FRESH_GATE7_INFORMATION_CEILING_DECOMPOSITION_EVIDENCE"
)

__all__ = [name for name in globals() if name.startswith("GATE7_")] + [
    "Gate7InformationCeilingWorld",
    "Gate7InformationCeilingBatchRanks",
    "Gate7InformationCeilingCheckpointRanks",
    "Gate7InformationCeilingRankSummary",
    "Gate7InformationCeilingPairedSummary",
    "generate_gate7_information_ceiling_world",
    "information_ceiling_world_batch",
    "load_verified_gate7_information_ceiling_checkpoint",
    "evaluate_information_ceiling_rank_batch",
    "aggregate_information_ceiling_rank_batches",
    "summarize_information_ceiling_ranks",
    "paired_information_ceiling_summary",
    "comparison_for_frozen_classifier",
    "classify_from_paired_summaries",
]
