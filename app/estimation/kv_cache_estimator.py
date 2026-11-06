"""
Pure, deterministic KV-cache memory estimator for autoregressive Transformer architectures.

Mathematical Formulation:
    bytes = 2 * batch_size * num_hidden_layers * num_key_value_heads * head_dim * total_cached_tokens * bytes_per_element

Explicitly separates model weight quantization (e.g., Q4_K_M, Q5_K_M) from runtime KV-cache tensor precision.
"""

from typing import Dict, List, Optional, Tuple
from app.core.config import settings
from app.schemas.request import (
    GPUTelemetry,
    KVCacheEstimateResult,
    KVDtype,
    ModelArchMetadata,
)

BYTES_PER_MIB = float(1024 * 1024)

# Bytes per element for supported KV-cache data types.
# Note: Quantized KV formats (q8_0, q4_0) in ggml have small block-scale overheads;
# we use nominal element byte sizes and document block-scale overhead in assumptions.
KV_DTYPE_BYTES_MAP: Dict[KVDtype, float] = {
    KVDtype.FP32: 4.0,
    KVDtype.FP16: 2.0,
    KVDtype.BF16: 2.0,
    KVDtype.FP8_E4M3: 1.0,
    KVDtype.FP8_E5M2: 1.0,
    KVDtype.INT8: 1.0,
    KVDtype.Q8_0: 1.0,
    KVDtype.Q4_0: 0.5,
}


def compute_kv_cache_bytes(
    *,
    num_hidden_layers: int,
    num_key_value_heads: int,
    head_dim: int,
    prompt_tokens: int,
    max_new_tokens: int,
    bytes_per_element: float,
    batch_size: int = 1,
    uses_sliding_window: bool = False,
    is_hybrid_or_nonstandard: bool = False,
) -> Dict[str, float | int]:
    """
    Pure mathematical function computing KV-cache memory usage in bytes and MiB.

    Formula:
        total_cached_tokens = batch_size * (prompt_tokens + max_new_tokens)
        bytes = 2 * num_hidden_layers * num_key_value_heads * head_dim * total_cached_tokens * bytes_per_element

    Raises:
        ValueError: If any dimensional input is non-positive or if architecture uses unsupported attention.
    """
    if num_hidden_layers <= 0:
        raise ValueError(f"num_hidden_layers must be > 0, got {num_hidden_layers}")
    if num_key_value_heads <= 0:
        raise ValueError(f"num_key_value_heads must be > 0, got {num_key_value_heads}")
    if head_dim <= 0:
        raise ValueError(f"head_dim must be > 0, got {head_dim}")
    if prompt_tokens <= 0:
        raise ValueError(f"prompt_tokens must be > 0, got {prompt_tokens}")
    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be > 0, got {max_new_tokens}")
    if bytes_per_element <= 0.0:
        raise ValueError(f"bytes_per_element must be > 0, got {bytes_per_element}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")

    if uses_sliding_window:
        raise ValueError(
            "Architecture uses sliding-window attention (SWA); linear full-sequence KV-cache formula is unsupported until SWA eviction is explicitly modeled."
        )
    if is_hybrid_or_nonstandard:
        raise ValueError(
            "Architecture uses hybrid/state-space or compressed latent attention (e.g., Mamba/MLA); standard 2*L*H_kv*D KV-cache formula is unsupported."
        )

    per_token_per_seq_bytes = int(
        round(2 * num_hidden_layers * num_key_value_heads * head_dim * bytes_per_element)
    )
    total_seq_tokens = prompt_tokens + max_new_tokens
    total_cached_tokens_across_batch = batch_size * total_seq_tokens

    prompt_phase_bytes = int(
        round(
            2
            * num_hidden_layers
            * num_key_value_heads
            * head_dim
            * (batch_size * prompt_tokens)
            * bytes_per_element
        )
    )
    generation_phase_bytes = int(
        round(
            2
            * num_hidden_layers
            * num_key_value_heads
            * head_dim
            * (batch_size * max_new_tokens)
            * bytes_per_element
        )
    )
    total_bytes = prompt_phase_bytes + generation_phase_bytes

    return {
        "per_token_kv_bytes": per_token_per_seq_bytes,
        "total_cached_tokens_per_seq": total_seq_tokens,
        "total_cached_tokens_across_batch": total_cached_tokens_across_batch,
        "prompt_phase_kv_bytes": prompt_phase_bytes,
        "prompt_phase_kv_mib": round(prompt_phase_bytes / BYTES_PER_MIB, 4),
        "generation_phase_kv_bytes": generation_phase_bytes,
        "generation_phase_kv_mib": round(generation_phase_bytes / BYTES_PER_MIB, 4),
        "total_predicted_kv_bytes": total_bytes,
        "total_predicted_kv_mib": round(total_bytes / BYTES_PER_MIB, 4),
    }


def evaluate_request_kv_cache(
    *,
    arch: ModelArchMetadata,
    prompt_tokens: Optional[int],
    max_new_tokens: int,
    declared_kv_dtype: Optional[KVDtype],
    batch_size: int,
    gpu_telemetry: Optional[GPUTelemetry] = None,
) -> Tuple[KVCacheEstimateResult, List[str]]:
    """
    High-level orchestrator that validates metadata/tokenizer readiness, applies
    KV-precision separation rules, invokes `compute_kv_cache_bytes`, and compares
    against available GPU VRAM.
    """
    warnings: List[str] = []

    if declared_kv_dtype is not None:
        effective_dtype = declared_kv_dtype
        explicitly_declared = True
    else:
        effective_dtype = KVDtype.FP16
        explicitly_declared = False
        warnings.append(
            "kv_dtype was not explicitly declared in request; defaulting to 'fp16' (2.0 bytes/element), "
            "which is Ollama's standard runtime KV-cache precision regardless of weight quantization."
        )

    dtype_bytes = KV_DTYPE_BYTES_MAP[effective_dtype]

    weight_quant_str = arch.weight_quantization_level or "unknown/unreported"
    weight_note = (
        f"Model weight quantization is '{weight_quant_str}', whereas KV-cache precision is modeled "
        f"independently as '{effective_dtype.value}' ({dtype_bytes} bytes/element). "
        "Weight quantization (e.g., Q4_K_M/Q5_K_M) compresses static model weights only and NEVER implies quantized KV-cache tensors."
    )

    assumptions = [
        "Standard dense or Grouped-Query Attention (GQA) with contiguous full-context KV retention.",
        "Factor of 2 accounts for separate Key (K) and Value (V) cache tensors per layer.",
        f"KV-cache precision is '{effective_dtype.value}' ({dtype_bytes} bytes/element), independent of weight quantization ('{weight_quant_str}').",
        "Does not include CUDA context overhead, activation workspace buffers, or ggml block-scale metadata overhead (~3-6% for q8_0/q4_0 KV caches).",
        "Assumes worst-case generation up to full `prompt_tokens + max_new_tokens` without early EOS termination.",
    ]

    if arch.uses_sliding_window:
        reason = "Model uses sliding-window attention (SWA); unsupported by linear full-history KV estimator."
        warnings.append(reason)
        return (
            KVCacheEstimateResult(
                estimation_possible=False,
                kv_dtype=effective_dtype,
                kv_dtype_bytes_per_element=dtype_bytes,
                kv_dtype_explicitly_declared=explicitly_declared,
                weight_quantization_note=weight_note,
                batch_size=batch_size,
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_new_tokens,
                assumptions=assumptions,
                unsupported_reason=reason,
            ),
            warnings,
        )

    if arch.is_hybrid_or_nonstandard:
        reason = f"Model architecture '{arch.architecture_family}' uses hybrid/non-standard attention layout; unsupported."
        warnings.append(reason)
        return (
            KVCacheEstimateResult(
                estimation_possible=False,
                kv_dtype=effective_dtype,
                kv_dtype_bytes_per_element=dtype_bytes,
                kv_dtype_explicitly_declared=explicitly_declared,
                weight_quantization_note=weight_note,
                batch_size=batch_size,
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_new_tokens,
                assumptions=assumptions,
                unsupported_reason=reason,
            ),
            warnings,
        )

    if not arch.metadata_available or any(
        x is None for x in (arch.num_hidden_layers, arch.num_key_value_heads, arch.head_dim)
    ):
        reason = "Architectural metadata (layers, kv_heads, head_dim) is unavailable; cannot compute KV-cache bytes."
        warnings.append(reason)
        return (
            KVCacheEstimateResult(
                estimation_possible=False,
                kv_dtype=effective_dtype,
                kv_dtype_bytes_per_element=dtype_bytes,
                kv_dtype_explicitly_declared=explicitly_declared,
                weight_quantization_note=weight_note,
                batch_size=batch_size,
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_new_tokens,
                assumptions=assumptions,
                unsupported_reason=reason,
            ),
            warnings,
        )

    if prompt_tokens is None or prompt_tokens <= 0:
        reason = "Exact prompt token count is unavailable; KV-cache estimation aborted rather than using character approximations."
        warnings.append(reason)
        return (
            KVCacheEstimateResult(
                estimation_possible=False,
                kv_dtype=effective_dtype,
                kv_dtype_bytes_per_element=dtype_bytes,
                kv_dtype_explicitly_declared=explicitly_declared,
                weight_quantization_note=weight_note,
                batch_size=batch_size,
                prompt_tokens=None,
                max_new_tokens=max_new_tokens,
                assumptions=assumptions,
                unsupported_reason=reason,
            ),
            warnings,
        )

    calc = compute_kv_cache_bytes(
        num_hidden_layers=int(arch.num_hidden_layers),  # type: ignore[arg-type]
        num_key_value_heads=int(arch.num_key_value_heads),  # type: ignore[arg-type]
        head_dim=int(arch.head_dim),  # type: ignore[arg-type]
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_new_tokens,
        bytes_per_element=dtype_bytes,
        batch_size=batch_size,
        uses_sliding_window=arch.uses_sliding_window,
        is_hybrid_or_nonstandard=arch.is_hybrid_or_nonstandard,
    )

    total_seq = int(calc["total_cached_tokens_per_seq"])
    if arch.context_length and total_seq > arch.context_length:
        warnings.append(
            f"Requested total sequence length ({total_seq} tokens) exceeds model's reported context_length ({arch.context_length})."
        )

    headroom_msg: Optional[str] = None
    total_kv_mib = float(calc["total_predicted_kv_mib"])
    if gpu_telemetry and gpu_telemetry.available and gpu_telemetry.free_vram_mib is not None:
        free_mib = gpu_telemetry.free_vram_mib
        usable_mib = max(0.0, free_mib - settings.vram_safety_margin_mib)
        if total_kv_mib > free_mib:
            headroom_msg = (
                f"CRITICAL: Predicted KV-cache ({total_kv_mib:.2f} MiB) exceeds total currently free VRAM ({free_mib:.2f} MiB)."
            )
            warnings.append(headroom_msg)
        elif total_kv_mib > usable_mib:
            headroom_msg = (
                f"WARNING: Predicted KV-cache ({total_kv_mib:.2f} MiB) fits in raw free VRAM ({free_mib:.2f} MiB) "
                f"but breaches the {settings.vram_safety_margin_mib:.0f} MiB safety margin."
            )
            warnings.append(headroom_msg)
        else:
            headroom_msg = (
                f"OK: Predicted KV-cache ({total_kv_mib:.2f} MiB) fits within free VRAM ({free_mib:.2f} MiB) "
                f"with safety margin ({settings.vram_safety_margin_mib:.0f} MiB)."
            )
    else:
        headroom_msg = "GPU VRAM telemetry unavailable; could not verify VRAM headroom."

    return (
        KVCacheEstimateResult(
            estimation_possible=True,
            kv_dtype=effective_dtype,
            kv_dtype_bytes_per_element=dtype_bytes,
            kv_dtype_explicitly_declared=explicitly_declared,
            weight_quantization_note=weight_note,
            batch_size=batch_size,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            total_cached_tokens_per_seq=int(calc["total_cached_tokens_per_seq"]),
            total_cached_tokens_across_batch=int(calc["total_cached_tokens_across_batch"]),
            prompt_phase_kv_bytes=int(calc["prompt_phase_kv_bytes"]),
            prompt_phase_kv_mib=float(calc["prompt_phase_kv_mib"]),
            generation_phase_kv_bytes=int(calc["generation_phase_kv_bytes"]),
            generation_phase_kv_mib=float(calc["generation_phase_kv_mib"]),
            total_predicted_kv_bytes=int(calc["total_predicted_kv_bytes"]),
            total_predicted_kv_mib=total_kv_mib,
            per_token_kv_bytes=int(calc["per_token_kv_bytes"]),
            vram_headroom_assessment=headroom_msg,
            assumptions=assumptions,
            unsupported_reason=None,
        ),
        warnings,
    )