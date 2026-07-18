"""Dashboard result adapters."""

from __future__ import annotations

from .base import ResultAdapter
from .step01_result import Step01ResultAdapter
from .step01_summary import Step01SummaryAdapter
from .step02_population import Step02PopulationAdapter


def default_adapters() -> list[ResultAdapter]:
    return [
        Step01ResultAdapter(),
        Step01SummaryAdapter(),
        Step02PopulationAdapter(),
    ]
