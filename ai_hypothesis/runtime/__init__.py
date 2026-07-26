"""Stable semantic contracts and local runtime primitives."""

from .allocation_outcomes import AllocationOutcome, AllocationOutcomeProjector, AttemptOutcome
from .context_views import PurposeContextRouter
from .contracts import (
    AttemptResult,
    AttemptStatus,
    EvidenceContribution,
    EvidenceDisposition,
    EvidenceDispositionKind,
    KnowledgeAssessment,
    KnowledgeAssessmentKind,
    KnowledgeDelta,
    LedgerEvent,
    ProjectedState,
    SchedulerAction,
    SchedulerDecision,
    WorkItem,
    WorkPurpose,
)
from .control import (
    ControlBatch,
    ControlStep,
    RuntimeControlLoop,
    WorkerSelectorV0,
    WorkPreparation,
    WorkPreparationBatch,
)
from .followups import FollowupMaterializer, FollowupRequest, FollowupSnapshot
from .graph_context import WorkGraphContextResolver
from .integration import (
    IntegrationBackpressureConfig,
    IntegrationBatch,
    IntegrationDisposition,
    IntegrationOverview,
    IntegrationSnapshot,
    IntegrationTracker,
    PendingEvidence,
)
from .integration_allocation_outcomes import (
    IntegrationAllocationOutcome,
    IntegrationAllocationOutcomeProjector,
    IntegrationAttemptOutcome,
    IntegrationWidthSummary,
    summarize_integration_allocations_by_width,
)
from .integration_parallelism import (
    IntegrationParallelismConfig,
    IntegrationPartitionAllocator,
    PartitionedBackpressureScheduler,
    PartitionedIntegrationContextRouter,
)
from .integration_partition_lineage import (
    HistoricalIntegrationPartition,
    PartitionAttemptLineage,
    PartitionedIntegrationLineage,
    PartitionedIntegrationLineageProjector,
    PartitionedIntegrationLineageSnapshot,
)
from .integration_partitions import (
    IntegrationPartition,
    IntegrationPartitionConfig,
    IntegrationPartitionPlan,
    IntegrationPartitionProjector,
    prepare_partition_integration_work,
)
from .integration_telemetry import (
    IntegrationBandwidthWindow,
    IntegrationTelemetryProjector,
    IntegrationTelemetrySnapshot,
)
from .integration_work import prepare_bounded_integration_work
from .knowledge import (
    KnowledgeRecord,
    KnowledgeSnapshot,
    KnowledgeStateProjector,
    KnowledgeStatus,
)
from .knowledge_verification import (
    KnowledgeVerificationConfig,
    KnowledgeVerificationOverview,
    KnowledgeVerificationTracker,
)
from .knowledge_work import prepare_bounded_knowledge_work
from .ledger import SQLiteResearchLedger
from .projector import ThreadStateProjector
from .scheduler import SchedulerConfig, SchedulerSignals, SchedulerV0, SchedulableThread
from .scheduler_trace import TracingScheduler, TracingSchedulerV0
from .scope_coverage import ScopeCoverageProjector, ScopeRegionCoverage, ThreadScopeCoverage
from .thread_consolidation import (
    ThreadConsolidationConfig,
    ThreadConsolidationPlan,
    ThreadConsolidationPlanner,
    ThreadConsolidationSource,
    prepare_thread_consolidation_work,
)
from .worker_runtime import AttemptRequest, WorkerAssignment, WorkerBank, WorkerRuntime

__all__ = [
    "AllocationOutcome",
    "AllocationOutcomeProjector",
    "AttemptOutcome",
    "AttemptRequest",
    "AttemptResult",
    "AttemptStatus",
    "ControlBatch",
    "ControlStep",
    "EvidenceContribution",
    "EvidenceDisposition",
    "EvidenceDispositionKind",
    "FollowupMaterializer",
    "FollowupRequest",
    "FollowupSnapshot",
    "HistoricalIntegrationPartition",
    "IntegrationAllocationOutcome",
    "IntegrationAllocationOutcomeProjector",
    "IntegrationAttemptOutcome",
    "IntegrationBackpressureConfig",
    "IntegrationBandwidthWindow",
    "IntegrationBatch",
    "IntegrationDisposition",
    "IntegrationOverview",
    "IntegrationParallelismConfig",
    "IntegrationPartition",
    "IntegrationPartitionAllocator",
    "IntegrationPartitionConfig",
    "IntegrationPartitionPlan",
    "IntegrationPartitionProjector",
    "IntegrationSnapshot",
    "IntegrationTelemetryProjector",
    "IntegrationTelemetrySnapshot",
    "IntegrationTracker",
    "IntegrationWidthSummary",
    "KnowledgeAssessment",
    "KnowledgeAssessmentKind",
    "KnowledgeDelta",
    "KnowledgeRecord",
    "KnowledgeSnapshot",
    "KnowledgeStateProjector",
    "KnowledgeStatus",
    "KnowledgeVerificationConfig",
    "KnowledgeVerificationOverview",
    "KnowledgeVerificationTracker",
    "LedgerEvent",
    "PartitionAttemptLineage",
    "PartitionedBackpressureScheduler",
    "PartitionedIntegrationContextRouter",
    "PartitionedIntegrationLineage",
    "PartitionedIntegrationLineageProjector",
    "PartitionedIntegrationLineageSnapshot",
    "PendingEvidence",
    "ProjectedState",
    "PurposeContextRouter",
    "RuntimeControlLoop",
    "SchedulerAction",
    "SchedulerConfig",
    "SchedulerDecision",
    "SchedulerSignals",
    "SchedulerV0",
    "SchedulableThread",
    "ScopeCoverageProjector",
    "ScopeRegionCoverage",
    "SQLiteResearchLedger",
    "ThreadConsolidationConfig",
    "ThreadConsolidationPlan",
    "ThreadConsolidationPlanner",
    "ThreadConsolidationSource",
    "ThreadScopeCoverage",
    "ThreadStateProjector",
    "TracingScheduler",
    "TracingSchedulerV0",
    "WorkerAssignment",
    "WorkerBank",
    "WorkerRuntime",
    "WorkerSelectorV0",
    "WorkGraphContextResolver",
    "WorkItem",
    "WorkPreparation",
    "WorkPreparationBatch",
    "WorkPurpose",
    "prepare_bounded_integration_work",
    "prepare_bounded_knowledge_work",
    "prepare_partition_integration_work",
    "prepare_thread_consolidation_work",
    "summarize_integration_allocations_by_width",
]
