"""FastAPI routes for dashboard API v1."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from .schemas import (
    ApiErrorV1,
    ExperimentDetailResponseV1,
    ExperimentListResponseV1,
    HealthResponseV1,
    IndexErrorListResponseV1,
    ReindexResponseV1,
    StatusResponseV1,
)
from .store import DashboardStore


def api_router(store: DashboardStore) -> APIRouter:
    router = APIRouter(prefix="/api/v1")

    @router.get("/health", response_model=HealthResponseV1)
    def health() -> HealthResponseV1:
        return HealthResponseV1()

    @router.get("/status", response_model=StatusResponseV1)
    def status() -> StatusResponseV1:
        return StatusResponseV1(status=store.snapshot().status)

    @router.get("/experiments", response_model=ExperimentListResponseV1)
    def experiments(
        research_step: str | None = None,
        experiment_type: str | None = None,
        seed: int | None = None,
        search: str | None = None,
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> ExperimentListResponseV1:
        items = store.snapshot().experiments
        if research_step:
            items = [item for item in items if item.identity.research_step == research_step]
        if experiment_type:
            items = [
                item
                for item in items
                if item.identity.experiment_type == experiment_type
            ]
        if seed is not None:
            items = [item for item in items if item.training.seed == seed]
        if search:
            needle = search.lower()
            items = [
                item
                for item in items
                if needle in item.identity.experiment_name.lower()
                or needle in item.identity.experiment_id.lower()
                or (
                    item.provenance.git_revision is not None
                    and needle in item.provenance.git_revision.lower()
                )
            ]
        total = len(items)
        return ExperimentListResponseV1(
            items=items[offset : offset + limit],
            total=total,
            limit=limit,
            offset=offset,
        )

    @router.get("/experiments/{experiment_id}", response_model=ExperimentDetailResponseV1)
    def experiment_detail(experiment_id: str) -> ExperimentDetailResponseV1:
        snapshot = store.snapshot()
        experiment = snapshot.experiments_by_id.get(experiment_id)
        if experiment is None:
            error = ApiErrorV1(
                code="EXPERIMENT_NOT_FOUND",
                message="Experiment was not found.",
            )
            raise HTTPException(status_code=404, detail=error.model_dump())
        return ExperimentDetailResponseV1(experiment=experiment)

    @router.get("/index-errors", response_model=IndexErrorListResponseV1)
    def index_errors() -> IndexErrorListResponseV1:
        items = store.snapshot().index_errors
        return IndexErrorListResponseV1(items=items, total=len(items))

    @router.post("/reindex", response_model=ReindexResponseV1)
    def reindex() -> ReindexResponseV1:
        return ReindexResponseV1(status=store.reindex().status)

    return router
