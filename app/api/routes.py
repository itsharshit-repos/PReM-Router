"""
FastAPI read-only profiling routes for PReM-Router Phase 1.
No LLM inference, routing, queuing, or model switching occurs here.
"""

from datetime import datetime, timezone
from typing import List
from fastapi import APIRouter
from app.core.config import fingerprint_prompt, logger, settings
from app.estimation.kv_cache_estimator import evaluate_request_kv_cache
from app.profiling.gpu_monitor import gpu_monitor
from app.profiling.model_profiler import model_profiler
from app.profiling.system_monitor import system_monitor
from app.profiling.tokenizer_profiler import tokenizer_profiler
from app.schemas.request import (
    GPUTelemetry,
    HealthResponse,
    ModelsListResponse,
    ProfileRequestPayload,
    ProfileRequestResponse,
    SystemResources,
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["system"])
def get_health() -> HealthResponse:
    """Returns service liveness and connectivity status for NVML and Ollama."""
    gpu_state = gpu_monitor.get_telemetry()
    ollama_up = model_profiler.check_ollama_reachable()
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=settings.app_version,
        nvml_available=gpu_state.available,
        ollama_reachable=ollama_up,
        timestamp_utc=datetime.now(timezone.utc),
    )


@router.get("/system/gpu", response_model=GPUTelemetry, tags=["telemetry"])
def get_system_gpu() -> GPUTelemetry:
    """Returns real-time NVIDIA GPU VRAM, utilization, and thermal metrics via NVML."""
    return gpu_monitor.get_telemetry()


@router.get("/system/resources", response_model=SystemResources, tags=["telemetry"])
def get_system_resources() -> SystemResources:
    """Returns host CPU, RAM, swap, and Ollama process metrics via psutil."""
    return system_monitor.get_system_resources()


@router.get("/models", response_model=ModelsListResponse, tags=["models"])
def get_models() -> ModelsListResponse:
    """Queries the local Ollama server for available models, VRAM residency, and architecture metadata."""
    return model_profiler.list_models()


@router.post("/profile/request", response_model=ProfileRequestResponse, tags=["profiling"])
def profile_request(payload: ProfileRequestPayload) -> ProfileRequestResponse:
    """
    Profiles a candidate LLM request without executing inference.
    Computes exact prompt token counts (where available), extracts model architecture metadata,
    predicts KV-cache footprint across prompt and generation phases, and snapshots current GPU VRAM.
    """
    logger.info(
        "Profiling request: model='%s', prompt=%s, max_new_tokens=%d, kv_dtype=%s, batch_size=%d",
        payload.model_name,
        fingerprint_prompt(payload.prompt),
        payload.max_new_tokens,
        payload.kv_dtype.value if payload.kv_dtype else "default(fp16)",
        payload.batch_size,
    )

    warnings: List[str] = []

    # 1. Exact Tokenization
    tok_result = tokenizer_profiler.count_prompt_tokens(
        model_name=payload.model_name,
        prompt=payload.prompt,
        tokenizer_override=payload.tokenizer_override,
    )
    if tok_result.error:
        warnings.append(tok_result.error)

    # 2. Model Architectural Metadata
    arch_meta = model_profiler.get_model_metadata(payload.model_name)
    warnings.extend(arch_meta.warnings)

    # 3. Real-time GPU Telemetry
    gpu_state = gpu_monitor.get_telemetry()
    if not gpu_state.available and gpu_state.error:
        warnings.append(f"GPU telemetry unavailable: {gpu_state.error}")

    # 4. Deterministic KV-Cache Estimation
    kv_estimate, kv_warnings = evaluate_request_kv_cache(
        arch=arch_meta,
        prompt_tokens=tok_result.prompt_tokens,
        max_new_tokens=payload.max_new_tokens,
        declared_kv_dtype=payload.kv_dtype,
        batch_size=payload.batch_size,
        gpu_telemetry=gpu_state,
    )
    warnings.extend(kv_warnings)

    # Deduplicate warnings while preserving order
    deduped_warnings = list(dict.fromkeys(warnings))

    return ProfileRequestResponse(
        model_name=payload.model_name,
        prompt_char_length=len(payload.prompt),
        tokenizer=tok_result,
        model_metadata=arch_meta,
        kv_cache_estimate=kv_estimate,
        gpu_telemetry=gpu_state,
        warnings=deduped_warnings,
        timestamp_utc=datetime.now(timezone.utc),
    )