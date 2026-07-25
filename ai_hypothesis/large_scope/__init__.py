"""Large-scope benchmarks built from frozen local Worker v1 semantics."""

from .evaluate import (
    ScopeEvaluation,
    ScopeWorkerMode,
    WindowEvidence,
    evaluate_scope_sample,
    evaluate_scope_widths,
)
from .metrics import (
    ScopeConditionSummary,
    ScopeMetricsAccumulator,
    summarize_scope_evaluations,
)
from .relevance import (
    LARGE_SCOPE_BENCHMARK_VERSION,
    LARGE_SCOPE_SPLIT_SEED_RANGES,
    LargeScopeRelevanceConfig,
    LargeScopeRelevanceSample,
    diverse_worker_indices,
    generate_large_scope_dataset,
    generate_large_scope_relevance,
    inspection_order,
    inspection_prefix,
    same_worker_indices,
)

__all__ = [
    "LARGE_SCOPE_BENCHMARK_VERSION",
    "LARGE_SCOPE_SPLIT_SEED_RANGES",
    "LargeScopeRelevanceConfig",
    "LargeScopeRelevanceSample",
    "ScopeConditionSummary",
    "ScopeEvaluation",
    "ScopeMetricsAccumulator",
    "ScopeWorkerMode",
    "WindowEvidence",
    "diverse_worker_indices",
    "evaluate_scope_sample",
    "evaluate_scope_widths",
    "generate_large_scope_dataset",
    "generate_large_scope_relevance",
    "inspection_order",
    "inspection_prefix",
    "same_worker_indices",
    "summarize_scope_evaluations",
]
