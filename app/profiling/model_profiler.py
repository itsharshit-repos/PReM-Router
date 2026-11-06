"""
Queries the local Ollama HTTP daemon for available models, active VRAM residency,
and GGUF architectural metadata (/api/tags, /api/ps, /api/show).
Never fabricates architectural numbers and explicitly tracks head_dim provenance.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import httpx
from app.core.config import KNOWN_MODEL_ARCHITECTURES, logger, settings
from app.schemas.request import (
    HeadDimSource,
    MetadataSource,
    ModelArchMetadata,
    ModelsListResponse,
    OllamaModelSummary,
)

BYTES_PER_MIB = float(1024 * 1024)


class ModelProfiler:
    """Read-only client for inspecting local Ollama models and architectural metadata."""

    def __init__(self, base_url: str = settings.ollama_base_url, timeout: float = settings.ollama_timeout_seconds) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def check_ollama_reachable(self) -> bool:
        """Pings Ollama /api/tags to verify daemon reachability."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def _get_running_models_map(self, client: httpx.Client) -> Dict[str, int]:
        """Queries GET /api/ps to check which models are currently loaded in VRAM."""
        loaded_vram: Dict[str, int] = {}
        try:
            resp = client.get(f"{self.base_url}/api/ps")
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("models", []):
                    name = item.get("name") or item.get("model")
                    size_vram = item.get("size_vram")
                    if name and isinstance(size_vram, int):
                        loaded_vram[name] = size_vram
        except Exception as exc:
            logger.debug("Could not query Ollama /api/ps: %s", exc)
        return loaded_vram

    @staticmethod
    def _extract_from_model_info(
        model_info: Dict[str, Any], details: Dict[str, Any]
    ) -> ModelArchMetadata:
        """
        Parses GGUF key-value metadata returned by Ollama's /api/show `model_info`.
        Handles arbitrary architecture prefixes (e.g., 'llama.', 'qwen2.') without guessing.
        """
        warnings: List[str] = []
        raw_keys_found: List[str] = []

        arch_family: Optional[str] = model_info.get("general.architecture")
        if arch_family:
            raw_keys_found.append("general.architecture")
        elif details.get("family"):
            arch_family = str(details.get("family"))

        def find_int_by_suffix(suffixes: List[str]) -> Tuple[Optional[int], Optional[str]]:
            if arch_family:
                for suffix in suffixes:
                    candidate = f"{arch_family}.{suffix}"
                    if candidate in model_info and isinstance(model_info[candidate], int):
                        return int(model_info[candidate]), candidate
            for k, v in model_info.items():
                for suffix in suffixes:
                    if k.endswith(f".{suffix}") and isinstance(v, int):
                        return int(v), k
            return None, None

        layers, k_layers = find_int_by_suffix(["block_count", "num_hidden_layers"])
        attn_heads, k_attn = find_int_by_suffix(["attention.head_count", "num_attention_heads"])
        kv_heads, k_kv = find_int_by_suffix(["attention.head_count_kv", "num_key_value_heads"])
        hidden_size, k_hidden = find_int_by_suffix(["embedding_length", "hidden_size"])
        explicit_k_dim, k_kdim = find_int_by_suffix(["attention.key_length", "head_dim"])
        explicit_v_dim, k_vdim = find_int_by_suffix(["attention.value_length"])
        ctx_len, k_ctx = find_int_by_suffix(["context_length"])
        sliding_win, k_sw = find_int_by_suffix(["attention.sliding_window", "sliding_window"])

        for k in (k_layers, k_attn, k_kv, k_hidden, k_kdim, k_vdim, k_ctx, k_sw):
            if k:
                raw_keys_found.append(k)

        # In Multi-Head Attention (MHA) where head_count_kv is omitted in GGUF, kv_heads == attn_heads
        if kv_heads is None and attn_heads is not None:
            kv_heads = attn_heads
            warnings.append(
                "attention.head_count_kv not explicitly present in model_info; defaulted to attention.head_count (MHA assumption)."
            )

        # Determine head_dim and its exact provenance
        head_dim: Optional[int] = None
        head_dim_source = HeadDimSource.UNAVAILABLE

        if explicit_k_dim is not None and explicit_k_dim > 0:
            if explicit_v_dim is not None and explicit_v_dim != explicit_k_dim:
                warnings.append(
                    f"Asymmetric key_length ({explicit_k_dim}) and value_length ({explicit_v_dim}) detected; standard estimator requires symmetric head_dim."
                )
            else:
                head_dim = explicit_k_dim
                head_dim_source = HeadDimSource.EXPLICIT_OLLAMA_METADATA
        elif hidden_size is not None and attn_heads is not None and attn_heads > 0:
            if hidden_size % attn_heads == 0:
                head_dim = hidden_size // attn_heads
                head_dim_source = HeadDimSource.DERIVED_HIDDEN_DIV_ATTN_HEADS
            else:
                warnings.append(
                    f"Cannot cleanly derive head_dim: hidden_size ({hidden_size}) is not divisible by num_attention_heads ({attn_heads})."
                )

        uses_sliding_window = bool(sliding_win is not None and sliding_win > 0)
        is_hybrid = False
        if arch_family and any(
            x in arch_family.lower() for x in ("mamba", "rwkv", "jamba", "hybrid", "deepseek2")
        ):
            is_hybrid = True
            warnings.append(
                f"Architecture family '{arch_family}' uses hybrid/state-space or MLA cache layouts not modeled by standard KV formula."
            )

        quant_level = details.get("quantization_level")
        param_size = details.get("parameter_size")

        core_complete = all(
            x is not None and x > 0 for x in (layers, kv_heads, head_dim)
        )
        if not core_complete:
            warnings.append(
                "Ollama /api/show response did not contain all required architectural dimensions (layers, kv_heads, head_dim)."
            )

        return ModelArchMetadata(
            metadata_available=core_complete,
            metadata_source=MetadataSource.OLLAMA_API_SHOW if core_complete else MetadataSource.UNAVAILABLE,
            architecture_family=arch_family,
            num_hidden_layers=layers,
            num_attention_heads=attn_heads,
            num_key_value_heads=kv_heads,
            hidden_size=hidden_size,
            head_dim=head_dim,
            head_dim_source=head_dim_source,
            context_length=ctx_len,
            weight_quantization_level=quant_level,
            parameter_size_label=param_size,
            uses_sliding_window=uses_sliding_window,
            is_hybrid_or_nonstandard=is_hybrid,
            raw_ollama_keys_found=raw_keys_found,
            warnings=warnings,
        )

    def _fallback_static_metadata(self, model_name: str, reason: str) -> ModelArchMetadata:
        """Checks the optional static registry when Ollama metadata is unavailable."""
        normalized = model_name.strip().lower()
        spec = KNOWN_MODEL_ARCHITECTURES.get(normalized)
        if spec is None:
            for key, candidate_spec in KNOWN_MODEL_ARCHITECTURES.items():
                if normalized.startswith(key):
                    spec = candidate_spec
                    break

        if not settings.enable_static_arch_fallback or spec is None:
            return ModelArchMetadata(
                metadata_available=False,
                metadata_source=MetadataSource.UNAVAILABLE,
                head_dim_source=HeadDimSource.UNAVAILABLE,
                warnings=[reason],
            )

        head_dim = spec.explicit_head_dim or (spec.hidden_size // spec.num_attention_heads)
        return ModelArchMetadata(
            metadata_available=True,
            metadata_source=MetadataSource.STATIC_FALLBACK_REGISTRY,
            architecture_family=spec.architecture_family,
            num_hidden_layers=spec.num_hidden_layers,
            num_attention_heads=spec.num_attention_heads,
            num_key_value_heads=spec.num_key_value_heads,
            hidden_size=spec.hidden_size,
            head_dim=head_dim,
            head_dim_source=HeadDimSource.STATIC_FALLBACK_REGISTRY,
            uses_sliding_window=spec.uses_sliding_window,
            is_hybrid_or_nonstandard=spec.is_hybrid_or_nonstandard,
            raw_ollama_keys_found=[],
            warnings=[
                f"{reason} Using verified static fallback registry specification for '{model_name}'."
            ],
        )

    def get_model_metadata(self, model_name: str) -> ModelArchMetadata:
        """
        Queries POST /api/show for a single model to retrieve architectural metadata.
        Falls back to static registry only if enabled and clearly marks provenance.
        """
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    f"{self.base_url}/api/show",
                    json={"name": model_name},
                )
                if resp.status_code != 200:
                    return self._fallback_static_metadata(
                        model_name,
                        f"Ollama /api/show returned HTTP {resp.status_code} for model '{model_name}'.",
                    )

                payload = resp.json()
                model_info = payload.get("model_info") or {}
                details = payload.get("details") or {}

                if not isinstance(model_info, dict) or not model_info:
                    return self._fallback_static_metadata(
                        model_name,
                        f"Ollama /api/show for '{model_name}' did not expose a 'model_info' dictionary.",
                    )

                extracted = self._extract_from_model_info(model_info, details)
                if not extracted.metadata_available and settings.enable_static_arch_fallback:
                    fallback = self._fallback_static_metadata(
                        model_name,
                        f"Incomplete 'model_info' in Ollama /api/show for '{model_name}'.",
                    )
                    if fallback.metadata_available:
                        fallback.weight_quantization_level = extracted.weight_quantization_level
                        fallback.parameter_size_label = extracted.parameter_size_label
                        return fallback
                return extracted

        except httpx.HTTPError as exc:
            logger.warning("Failed to query Ollama /api/show for %s: %s", model_name, exc)
            return self._fallback_static_metadata(
                model_name,
                f"Ollama daemon unreachable at {self.base_url} ({str(exc)}).",
            )

    def list_models(self) -> ModelsListResponse:
        """Queries GET /api/tags and enriches each model with /api/ps and /api/show metadata."""
        now = datetime.now(timezone.utc)
        try:
            with httpx.Client(timeout=self.timeout) as client:
                tags_resp = client.get(f"{self.base_url}/api/tags")
                if tags_resp.status_code != 200:
                    return ModelsListResponse(
                        ollama_reachable=False,
                        ollama_base_url=self.base_url,
                        error=f"GET /api/tags returned HTTP {tags_resp.status_code}",
                        models=[],
                        timestamp_utc=now,
                    )

                running_map = self._get_running_models_map(client)
                raw_models = tags_resp.json().get("models", [])
                summaries: List[OllamaModelSummary] = []

                for m in raw_models:
                    name = str(m.get("name") or m.get("model") or "unknown")
                    size_b = m.get("size")
                    size_mib = round(size_b / BYTES_PER_MIB, 2) if isinstance(size_b, int) else None
                    vram_b = running_map.get(name)
                    arch = self.get_model_metadata(name)

                    summaries.append(
                        OllamaModelSummary(
                            name=name,
                            size_bytes=size_b if isinstance(size_b, int) else None,
                            size_mib=size_mib,
                            digest=m.get("digest"),
                            modified_at=m.get("modified_at"),
                            currently_loaded_in_vram=name in running_map,
                            loaded_vram_bytes=vram_b,
                            loaded_vram_mib=round(vram_b / BYTES_PER_MIB, 2) if vram_b is not None else None,
                            architecture=arch,
                        )
                    )

                return ModelsListResponse(
                    ollama_reachable=True,
                    ollama_base_url=self.base_url,
                    error=None,
                    models=summaries,
                    timestamp_utc=now,
                )

        except httpx.HTTPError as exc:
            err = f"Could not connect to Ollama at {self.base_url}: {str(exc)}"
            logger.warning(err)
            return ModelsListResponse(
                ollama_reachable=False,
                ollama_base_url=self.base_url,
                error=err,
                models=[],
                timestamp_utc=now,
            )


model_profiler = ModelProfiler()