"""Large-scope benchmarks built from frozen local Worker v1 semantics."""

from .coverage_planner import CoverageAwareScopePlanner, CoverageScopePlan
from .evaluate import (
    ScopeEvaluation,
    ScopeWorkerMode,
    WindowEvidence,
    evaluate_scope_batch,
    evaluate_scope_sample,
    evaluate_scope_widths,
)
from .metrics import (
    ScopeConditionSummary,
    ScopeMetricsAccumulator,
    summarize_scope_evaluations,
)
from .paired_metrics import ScopePairedMetricsAccumulator, ScopePairedSummary
from .persistent_experiment import (
    PersistentScopeEvaluation,
    PersistentScopeEvaluationProjector,
    PersistentScopeExperiment,
    PersistentScopeWorkerSelector,
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
from .result_audit import (
    LargeScopeResultAudit,
    ResultAuditIssue,
    audit_large_scope_result,
    render_large_scope_audit_markdown,
)
from .runtime_bridge import (
    FixedScopeScheduler,
    LargeScopeRuntimeContextProvider,
    LargeScopeRuntimeWorkerBank,
    PlannedScopeWorkerSelector,
    large_scope_region_id,
    large_scope_worker_id,
    planned_scope_worker_selector,
)

__all__ = [
    "CoverageAwareScopePlanner",
    "CoverageScopePlan",
    "FixedScopeScheduler",
    "LARGE_SCOPE_BENCHMARK_VERSION",
    "LARGE_SCOPE_SPLIT_SEED_RANGES",
    "LargeScopeRelevanceConfig",
    "LargeScopeRelevanceSample",
    "LargeScopeResultAudit",
    "LargeScopeRuntimeContextProvider",
    "LargeScopeRuntimeWorkerBank",
    "PersistentScopeEvaluation",
    "PersistentScopeEvaluationProjector",
    "PersistentScopeExperiment",
    "PersistentScopeWorkerSelector",
    "PlannedScopeWorkerSelector",
    "ResultAuditIssue",
    "ScopeConditionSummary",
    "ScopeEvaluation",
    "ScopeMetricsAccumulator",
    "ScopePairedMetricsAccumulator",
    "ScopePairedSummary",
    "ScopeWorkerMode",
    "WindowEvidence",
    "audit_large_scope_result",
    "diverse_worker_indices",
    "evaluate_scope_batch",
    "evaluate_scope_sample",
    "evaluate_scope_widths",
    "generate_large_scope_dataset",
    "generate_large_scope_relevance",
    "inspection_order",
    "inspection_prefix",
    "large_scope_region_id",
    "large_scope_worker_id",
    "planned_scope_worker_selector",
    "render_large_scope_audit_markdown",
    "same_worker_indices",
    "summarize_scope_evaluations",
]
