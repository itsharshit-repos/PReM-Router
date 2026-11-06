"""
Pydantic v2 data contracts for PReM-Router Phase 1 endpoints and internal profiling objects.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class KVDtype(str, Enum):
    """Explicit KV-cache element data types. Never inferred from model weight quantization."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    INT8 = "int8"
    Q8_0 = "q8_0"
    Q4_0 = "q4_0"


class TokenizerStatus(str, Enum):
    """Status of exact tokenization for a request."""
    EXACT_HF = "exact_hf"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


class HeadDimSource(str, Enum):
    """Provenance of the head_dim architectural parameter."""
    EXPLICIT_OLLAMA_METADATA = "explicit_ollama_metadata"
    DERIVED_HIDDEN_DIV_ATTN_HEADS = "derived_hidden_size_div_num_attention_heads"
    STATIC_FALLBACK_REGISTRY = "static_fallback_registry"
    UNAVAILABLE = "unavailable"


class MetadataSource(str, Enum):
    """Provenance of the overall model architectural metadata."""
    OLLAMA_API_SHOW = "ollama_api_show"
    STATIC_FALLBACK_REGISTRY = "static_fallback_registry"
    UNAVAILABLE = "unavailable"


class GPUTelemetry(BaseModel):
    """Real-time GPU snapshot from NVML. No fabricated defaults when unavailable."""
    available: bool = Field(..., description="True if NVML initialized and queried the GPU")
    device_index: int = Field(..., description="Queried NVML device index")
    device_name: Optional[str] = None
    driver_version: Optional[str] = None
    nvml_version: Optional[str] = None
    total_vram_bytes: Optional[int] = None
    used_vram_bytes: Optional[int] = None
    free_vram_bytes: Optional[int] = None
    total_vram_mib: Optional[float] = None
    used_vram_mib: Optional[float] = None
    free_vram_mib: Optional[float] = None
    gpu_utilization_pct: Optional[float] = None
    memory_utilization_pct: Optional[float] = None
    temperature_celsius: Optional[int] = None
    power_usage_watts: Optional[float] = None
    error: Optional[str] = None
    timestamp_utc: datetime


class OllamaProcessInfo(BaseModel):
    """Process-level telemetry for running Ollama host/runner instances."""
    pid: int
    name: str
    rss_bytes: int
    rss_mib: float
    cpu_percent: float
    status: str


class SystemResources(BaseModel):
    """Host CPU, RAM, swap, and Ollama process snapshot via psutil."""
    cpu_physical_cores: Optional[int] = None
    cpu_logical_cores: Optional[int] = None
    cpu_usage_pct: float
    load_avg_1m_5m_15m: Optional[List[float]] = None
    ram_total_bytes: int
    ram_available_bytes: int
    ram_used_bytes: int
    ram_total_mib: float
    ram_available_mib: float
    ram_used_mib: float
    ram_usage_pct: float
    swap_total_mib: float
    swap_used_mib: float
    swap_usage_pct: float
    ollama_processes: List[OllamaProcessInfo] = Field(default_factory=list)
    timestamp_utc: datetime


class ModelArchMetadata(BaseModel):
    """Extracted architectural parameters required for KV-cache sizing."""
    metadata_available: bool
    metadata_source: MetadataSource
    architecture_family: Optional[str] = None
    num_hidden_layers: Optional[int] = None
    num_attention_heads: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    hidden_size: Optional[int] = None
    head_dim: Optional[int] = None
    head_dim_source: HeadDimSource = HeadDimSource.UNAVAILABLE
    context_length: Optional[int] = None
    weight_quantization_level: Optional[str] = Field(
        default=None,
        description="Weight quantization reported by Ollama (e.g., Q4_K_M). Distinct from KV-cache precision.",
    )
    parameter_size_label: Optional[str] = None
    uses_sliding_window: bool = False
    is_hybrid_or_nonstandard: bool = False
    raw_ollama_keys_found: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class OllamaModelSummary(BaseModel):
    """Summary of a model discovered on the local Ollama server."""
    name: str
    size_bytes: Optional[int] = None
    size_mib: Optional[float] = None
    digest: Optional[str] = None
    modified_at: Optional[str] = None
    currently_loaded_in_vram: bool = False
    loaded_vram_bytes: Optional[int] = None
    loaded_vram_mib: Optional[float] = None
    architecture: ModelArchMetadata


class ModelsListResponse(BaseModel):
    """Response schema for GET /models."""
    ollama_reachable: bool
    ollama_base_url: str
    error: Optional[str] = None
    models: List[OllamaModelSummary] = Field(default_factory=list)
    timestamp_utc: datetime


class TokenizerResult(BaseModel):
    """Result of attempting exact tokenization on a prompt."""
    status: TokenizerStatus
    tokenizer_identifier: Optional[str] = None
    prompt_tokens: Optional[int] = None
    error: Optional[str] = None


class KVCacheEstimateResult(BaseModel):
    """Output of the deterministic KV-cache memory estimator."""
    estimation_possible: bool
    kv_dtype: KVDtype
    kv_dtype_bytes_per_element: float
    kv_dtype_explicitly_declared: bool
    weight_quantization_note: str
    batch_size: int
    prompt_tokens: Optional[int] = None
    max_new_tokens: int
    total_cached_tokens_per_seq: Optional[int] = None
    total_cached_tokens_across_batch: Optional[int] = None
    prompt_phase_kv_bytes: Optional[int] = None
    prompt_phase_kv_mib: Optional[float] = None
    generation_phase_kv_bytes: Optional[int] = None
    generation_phase_kv_mib: Optional[float] = None
    total_predicted_kv_bytes: Optional[int] = None
    total_predicted_kv_mib: Optional[float] = None
    per_token_kv_bytes: Optional[int] = None
    vram_headroom_assessment: Optional[str] = None
    assumptions: List[str] = Field(default_factory=list)
    unsupported_reason: Optional[str] = None


class ProfileRequestPayload(BaseModel):
    """Input payload for POST /profile/request."""
    model_name: str = Field(..., min_length=1, examples=["llama3.2:1b", "qwen2.5:3b"])
    prompt: str = Field(..., min_length=1, description="Input prompt text to tokenize and profile")
    max_new_tokens: int = Field(default=256, gt=0, le=131072)
    kv_dtype: Optional[KVDtype] = Field(
        default=None,
        description="Optional declared KV-cache dtype. Defaults to fp16 with a warning if omitted.",
    )
    batch_size: int = Field(default=1, gt=0, le=512)
    tokenizer_override: Optional[str] = Field(
        default=None,
        description="Optional Hugging Face repo or local path override for exact tokenization.",
    )


class ProfileRequestResponse(BaseModel):
    """Comprehensive read-only profiling response for POST /profile/request."""
    model_name: str
    prompt_char_length: int
    tokenizer: TokenizerResult
    model_metadata: ModelArchMetadata
    kv_cache_estimate: KVCacheEstimateResult
    gpu_telemetry: GPUTelemetry
    warnings: List[str] = Field(default_factory=list)
    timestamp_utc: datetime


class HealthResponse(BaseModel):
    """Response schema for GET /health."""
    status: str
    service: str
    version: str
    nvml_available: bool
    ollama_reachable: bool
    timestamp_utc: datetime