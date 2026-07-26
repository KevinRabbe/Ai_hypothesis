"""Coverage-aware large-scope planning backed by incremental scope materialization."""

from __future__ import annotations

from ai_hypothesis.runtime.scope_coverage_index import SQLiteIndexedScopeCoverage

from .coverage_planner import CoverageAwareScopePlanner
from .evaluate import ScopeWorkerMode
from .relevance import LargeScopeRelevanceSample


class IndexedCoverageAwareScopePlanner(CoverageAwareScopePlanner):
    """Preserve coverage-planning policy while replacing full-ledger replay queries."""

    def __init__(
        self,
        coverage: SQLiteIndexedScopeCoverage,
        sample: LargeScopeRelevanceSample,
        mode: ScopeWorkerMode | str,
    ) -> None:
        if coverage.ledger is None:
            raise ValueError("indexed scope coverage must expose its Research Ledger")
        self.coverage_index = coverage
        super().__init__(coverage.ledger, sample, mode)

    def coverage_for(self, thread_id: str):
        return self.coverage_index.for_thread(thread_id)
