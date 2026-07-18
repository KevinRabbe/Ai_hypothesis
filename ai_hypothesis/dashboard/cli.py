"""Command-line entry point for the local dashboard server."""

from __future__ import annotations

import argparse

import uvicorn

from .app import create_app
from .settings import make_settings


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AI Hypothesis dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--results-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    settings = make_settings(
        host=args.host,
        port=args.port,
        results_dir=args.results_dir,
    )
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port)
