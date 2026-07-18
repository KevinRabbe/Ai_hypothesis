"""Typed dashboard errors used by indexing and API boundaries."""

from __future__ import annotations


class DashboardError(Exception):
    """Base class for expected dashboard-domain errors."""


class ArtifactParseError(DashboardError):
    """An artifact could not be parsed as JSON."""


class ArtifactValidationError(DashboardError):
    """An artifact parsed but did not satisfy a supported contract."""


class AdapterAmbiguityError(DashboardError):
    """More than one adapter claimed an artifact without a safe precedence."""


class DuplicateConflictError(DashboardError):
    """Two artifacts share an experiment identity but conflict scientifically."""


class ExperimentNotFoundError(DashboardError):
    """A requested experiment id is not present in the current index."""


class ResultsDirectoryAccessError(DashboardError):
    """The configured results directory could not be inspected."""
