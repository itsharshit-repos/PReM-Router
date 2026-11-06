"""
API integration tests for FastAPI endpoints using mocked GPU, Ollama, and Tokenizer backends.
"""

from datetime import datetime, timezone
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app
from app.schemas.request import (
    GPUTelemetry,
    HeadDimSource,
    MetadataSource,
    ModelArchMetadata,
    TokenizerResult,
    TokenizerStatus,
)

client = TestClient(app)


def test_health_endpoint() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "nvml_available" in body
    assert "ollama_reachable" in body


def test_profile_request_endpoint_with_mocks() -> None:
    """Tests POST /profile/request end-to-end with mocked tokenizer, model metadata, and GPU."""
    mock_tok = TokenizerResult(
        status=TokenizerStatus.EXACT_HF,
        tokenizer_identifier="meta-llama/Llama-3.2-1B",
        prompt_tokens=64,
        error=None,
    )
    mock_arch = ModelArchMetadata(
        metadata_available=True,
        metadata_source=MetadataSource.OLLAMA_API_SHOW,
        architecture_family="llama",
        num_hidden_layers=16,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_size=2048,
        head_dim=64,
        head_dim_source=HeadDimSource.EXPLICIT_OLLAMA_METADATA,
        context_length=131072,
        weight_quantization_level="Q4_K_M",
    )
    mock_gpu = GPUTelemetry(
        available=True,
        device_index=0,
        device_name="NVIDIA GeForce RTX 3050 6GB Laptop GPU",
        total_vram_bytes=6442450944,
        used_vram_bytes=2147483648,
        free_vram_bytes=4294967296,
        total_vram_mib=6144.0,
        used_vram_mib=2048.0,
        free_vram_mib=4096.0,
        gpu_utilization_pct=12.0,
        temperature_celsius=51,
        timestamp_utc=datetime.now(timezone.utc),
    )

    with patch(
        "app.api.routes.tokenizer_profiler.count_prompt_tokens", return_value=mock_tok
    ), patch(
        "app.api.routes.model_profiler.get_model_metadata", return_value=mock_arch
    ), patch(
        "app.api.routes.gpu_monitor.get_telemetry", return_value=mock_gpu
    ):
        resp = client.post(
            "/profile/request",
            json={
                "model_name": "llama3.2:1b",
                "prompt": "Benchmark test prompt",
                "max_new_tokens": 448,
                "kv_dtype": "fp16",
                "batch_size": 1,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["tokenizer"]["status"] == "exact_hf"
        assert data["tokenizer"]["prompt_tokens"] == 64
        assert data["kv_cache_estimate"]["total_cached_tokens_per_seq"] == 512
        # 512 tokens * 32,768 bytes/token = 16,777,216 bytes (16.0 MiB)
        assert data["kv_cache_estimate"]["total_predicted_kv_bytes"] == 16777216
        assert data["kv_cache_estimate"]["total_predicted_kv_mib"] == 16.0
        assert data["gpu_telemetry"]["free_vram_mib"] == 4096.0


def test_profile_request_when_tokenizer_unavailable() -> None:
    """Verifies that if exact tokenization is unavailable, KV estimation safely aborts instead of guessing."""
    mock_tok = TokenizerResult(
        status=TokenizerStatus.UNAVAILABLE,
        tokenizer_identifier=None,
        prompt_tokens=None,
        error="Exact tokenizer not found in local cache.",
    )
    with patch("app.api.routes.tokenizer_profiler.count_prompt_tokens", return_value=mock_tok):
        resp = client.post(
            "/profile/request",
            json={
                "model_name": "unknown-model:latest",
                "prompt": "Do not estimate my tokens using len(prompt) // 4",
                "max_new_tokens": 256,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["tokenizer"]["status"] == "unavailable"
        assert data["tokenizer"]["prompt_tokens"] is None
        assert data["kv_cache_estimate"]["estimation_possible"] is False
        assert data["kv_cache_estimate"]["total_predicted_kv_bytes"] is None