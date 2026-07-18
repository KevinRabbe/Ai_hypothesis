"""FastAPI application factory for the local dashboard."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .api import api_router
from .settings import DashboardSettings
from .store import DashboardStore


def create_app(settings: DashboardSettings) -> FastAPI:
    store = DashboardStore(results_dir=settings.results_dir)
    app = FastAPI(title="AI Hypothesis Research Dashboard")
    app.include_router(api_router(store))

    dist = settings.frontend_dist
    if dist is not None and dist.is_dir() and (dist / "index.html").is_file():
        assets = dist / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def frontend(path: str = ""):
            index = Path(dist) / "index.html"
            return HTMLResponse(index.read_text(encoding="utf-8"))

    else:

        @app.get("/", include_in_schema=False)
        def missing_frontend() -> HTMLResponse:
            return HTMLResponse(
                "<h1>AI Hypothesis Research Dashboard API</h1>"
                "<p>Frontend build assets are not present. Run the Vite dev server "
                "or build dashboard/frontend.</p>"
            )

    return app
