"""
Configuration and structured logging setup for PReM-Router Phase 1.
Binds strictly to 127.0.0.1 by default and enforces prompt redaction in logs.
"""

import hashlib
import logging
import sys
from typing import Dict, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class StaticArchSpec(BaseModel):
    """Fallback verified architecture specification when Ollama /api/show is incomplete."""
    architecture_family: str
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    hidden_size: int
    explicit_head_dim: Optional[int] = None
    uses_sliding_window: bool = False
    is_hybrid_or_nonstandard: bool = False


# Verified architectural specifications for Phase 1 target models (Llama 3.2 1B & Qwen 2.5 3B)
# Used ONLY when Ollama GGUF metadata cannot be queried or lacks specific tensor dimensions,
# and explicitly marked with provenance source="static_fallback_registry".
KNOWN_MODEL_ARCHITECTURES: Dict[str, StaticArchSpec] = {
    "llama3.2:1b": StaticArchSpec(
        architecture_family="llama",
        num_hidden_layers=16,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_size=2048,
        explicit_head_dim=64,
        uses_sliding_window=False,
        is_hybrid_or_nonstandard=False,
    ),
    "qwen2.5:3b": StaticArchSpec(
        architecture_family="qwen2",
        num_hidden_layers=36,
        num_attention_heads=16,
        num_key_value_heads=2,
        hidden_size=2048,
        explicit_head_dim=128,
        uses_sliding_window=False,
        is_hybrid_or_nonstandard=False,
    ),
}

# Default mapping from Ollama model prefixes/tags to Hugging Face tokenizer repositories.
# Can be overridden via environment variable PREM_TOKENIZER_MAP.
DEFAULT_TOKENIZER_MAP: Dict[str, str] = {
    "llama3.2:1b": "meta-llama/Llama-3.2-1B",
    "llama3.2:1b-instruct-q4_K_M": "meta-llama/Llama-3.2-1B-Instruct",
    "qwen2.5:3b": "Qwen/Qwen2.5-3B",
    "qwen2.5:3b-instruct": "Qwen/Qwen2.5-3B-Instruct",
}


class Settings(BaseSettings):
    """Application settings loaded from environment variables with PREM_ prefix."""

    app_name: str = "PReM-Router Phase 1 Profiler"
    app_version: str = "0.1.0"
    bind_host: str = Field(default="127.0.0.1", description="Localhost bind address")
    bind_port: int = Field(default=8000, ge=1024, le=65535)

    ollama_base_url: str = Field(
        default="http://127.0.0.1:11434",
        description="Base URL of the local running Ollama server",
    )
    ollama_timeout_seconds: float = Field(default=5.0, gt=0.0)

    gpu_device_index: int = Field(default=0, ge=0)
    vram_safety_margin_mib: float = Field(
        default=512.0,
        ge=0.0,
        description="Safety buffer in MiB when evaluating free VRAM headroom on 6GB RTX 3050",
    )

    hf_token: Optional[str] = Field(
        default=None,
        description="Optional Hugging Face token from environment (never hardcoded) for gated tokenizers like Llama 3.2",
    )
    tokenizer_local_files_only: bool = Field(
        default=False,
        description="If True, only load tokenizers from local Hugging Face cache without network calls",
    )
    tokenizer_map: Dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_TOKENIZER_MAP))
    enable_static_arch_fallback: bool = Field(
        default=True,
        description="Allow fallback to verified static architectural specs if Ollama /api/show is unreachable",
    )

    log_level: str = Field(default="INFO")
    log_redact_prompts: bool = Field(
        default=True,
        description="Never log raw prompt content when True; log only SHA-256 prefix and char length",
    )

    model_config = SettingsConfigDict(
        env_prefix="PREM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()


def setup_logging() -> logging.Logger:
    """Configures structured key-value logging to stdout."""
    logger = logging.getLogger("prem_router")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    logger.propagate = False
    return logger


logger = setup_logging()


def fingerprint_prompt(prompt: str) -> str:
    """
    Produces a safe log representation of a prompt.
    Never exposes raw prompt text if settings.log_redact_prompts is True.
    """
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
    if settings.log_redact_prompts:
        return f"<redacted:sha256={digest},chars={len(prompt)}>"
    preview = prompt[:32].replace("\n", "\\n")
    return f"<raw_preview='{preview}...',sha256={digest},chars={len(prompt)}>"