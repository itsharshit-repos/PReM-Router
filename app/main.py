"""
Main FastAPI entrypoint for PReM-Router Phase 1.
Binds strictly to 127.0.0.1 for local development security.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
import uvicorn
from fastapi import FastAPI
from app.api.routes import router as api_router
from app.core.config import logger, settings
from app.profiling.gpu_monitor import gpu_monitor


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manages NVML initialization and clean shutdown across the FastAPI lifespan."""
    logger.info("Starting %s v%s on %s:%d", settings.app_name, settings.app_version, settings.bind_host, settings.bind_port)
    gpu_monitor.initialize()
    yield
    gpu_monitor.shutdown()
    logger.info("Shutting down %s", settings.app_name)


def create_app() -> FastAPI:
    """Application factory for PReM-Router Phase 1."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Phase 1 read-only hardware, model metadata, and deterministic KV-cache profiling service "
            "for local LLM inference on resource-constrained GPUs (RTX 3050 6GB)."
        ),
        lifespan=lifespan,
    )
    app.include_router(api_router)
    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.bind_host,
        port=settings.bind_port,
        reload=False,
        log_level=settings.log_level.lower(),
    )