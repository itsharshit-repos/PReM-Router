"""
Unit tests for deterministic KV-cache memory estimation and weight-vs-KV precision separation.
"""

import pytest
from app.estimation.kv_cache_estimator import compute_kv_cache_bytes, evaluate_request_kv_cache
from app.schemas.request import (
    HeadDimSource,
    KVDtype,
    MetadataSource,
    ModelArchMetadata,
)


def test_llama_3_2_1b_exact_kv_math() -> None:
    """
    Verifies exact deterministic math for Llama 3.2 1B:
    layers=16, kv_heads=8, head_dim=64, fp16 (2 bytes).
    Per token per seq = 2 * 16 * 8 * 64 * 2 = 32,768 bytes (32 KiB / token).
    For 512 prompt + 512 new = 1024 tokens -> 33,554,432 bytes (32.0 MiB).
    """
    res = compute_kv_cache_bytes(
        num_hidden_layers=16,
        num_key_value_heads=8,
        head_dim=64,
        prompt_tokens=512,
        max_new_tokens=512,
        bytes_per_element=2.0,
        batch_size=1,
    )
    assert res["per_token_kv_bytes"] == 32768
    assert res["total_cached_tokens_per_seq"] == 1024
    assert res["total_predicted_kv_bytes"] == 33554432
    assert res["total_predicted_kv_mib"] == 32.0


def test_qwen_2_5_3b_exact_kv_math() -> None:
    """
    Verifies exact deterministic math for Qwen 2.5 3B:
    layers=36, kv_heads=2, head_dim=128, fp16 (2 bytes).
    Per token per seq = 2 * 36 * 2 * 128 * 2 = 36,864 bytes (36 KiB / token).
    For 1024 tokens -> 37,748,736 bytes (36.0 MiB).
    """
    res = compute_kv_cache_bytes(
        num_hidden_layers=36,
        num_key_value_heads=2,
        head_dim=128,
        prompt_tokens=256,
        max_new_tokens=768,
        bytes_per_element=2.0,
        batch_size=1,
    )
    assert res["per_token_kv_bytes"] == 36864
    assert res["total_predicted_kv_bytes"] == 37748736
    assert res["total_predicted_kv_mib"] == 36.0


def test_invalid_inputs_rejected() -> None:
    """Ensures non-positive dimensions or token counts raise ValueError."""
    with pytest.raises(ValueError, match="prompt_tokens must be > 0"):
        compute_kv_cache_bytes(
            num_hidden_layers=16,
            num_key_value_heads=8,
            head_dim=64,
            prompt_tokens=0,
            max_new_tokens=128,
            bytes_per_element=2.0,
        )


def test_sliding_window_and_hybrid_rejected() -> None:
    """Ensures sliding-window attention and hybrid architectures are flagged as unsupported."""
    with pytest.raises(ValueError, match="sliding-window attention"):
        compute_kv_cache_bytes(
            num_hidden_layers=16,
            num_key_value_heads=8,
            head_dim=64,
            prompt_tokens=100,
            max_new_tokens=100,
            bytes_per_element=2.0,
            uses_sliding_window=True,
        )


def test_weight_quantization_never_determines_kv_dtype() -> None:
    """
    Ensures a Q4_K_M weight-quantized model defaults to FP16 KV-cache (2.0 bytes)
    when kv_dtype is omitted, and emits an explicit warning.
    """
    arch = ModelArchMetadata(
        metadata_available=True,
        metadata_source=MetadataSource.OLLAMA_API_SHOW,
        architecture_family="llama",
        num_hidden_layers=16,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_size=2048,
        head_dim=64,
        head_dim_source=HeadDimSource.EXPLICIT_OLLAMA_METADATA,
        weight_quantization_level="Q4_K_M",
    )
    est, warnings = evaluate_request_kv_cache(
        arch=arch,
        prompt_tokens=100,
        max_new_tokens=100,
        declared_kv_dtype=None,
        batch_size=1,
    )
    assert est.estimation_possible is True
    assert est.kv_dtype == KVDtype.FP16
    assert est.kv_dtype_bytes_per_element == 2.0
    assert est.kv_dtype_explicitly_declared is False
    assert "Q4_K_M" in est.weight_quantization_note
    assert any("defaulting to 'fp16'" in w for w in warnings)