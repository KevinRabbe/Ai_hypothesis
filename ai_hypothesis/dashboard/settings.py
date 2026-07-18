"""Dashboard runtime settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DashboardSettings:
    host: str = "127.0.0.1"
    port: int = 8000
    repo_root: Path = Path.cwd()
    results_dir: Path = Path.cwd() / "results"
    frontend_dist: Path | None = None


def discover_repo_root() -> Path:
    package_path = Path(__file__).resolve()
    for parent in package_path.parents:
        if (parent / "ai_hypothesis").is_dir() and (parent / "README.md").is_file():
            return parent
    return package_path.parents[2]


def make_settings(
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    results_dir: str | None = None,
) -> DashboardSettings:
    repo_root = discover_repo_root()
    resolved_results = Path(results_dir).resolve() if results_dir else repo_root / "results"
    frontend_dist = repo_root / "dashboard" / "frontend" / "dist"
    return DashboardSettings(
        host=host,
        port=port,
        repo_root=repo_root,
        results_dir=resolved_results,
        frontend_dist=frontend_dist,
    )
