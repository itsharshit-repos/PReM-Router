"""
Exact tokenizer profiling module using Hugging Face `transformers.AutoTokenizer`.
Strictly prohibits silent character-count or word-count heuristic approximations.
"""

from typing import Any, Dict, Optional
from app.core.config import fingerprint_prompt, logger, settings
from app.schemas.request import TokenizerResult, TokenizerStatus

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoTokenizer = None  # type: ignore
    TRANSFORMERS_AVAILABLE = False


class TokenizerProfiler:
    """
    Resolves and caches Hugging Face tokenizers corresponding to Ollama model names.
    Returns explicit UNAVAILABLE or ERROR statuses if an exact tokenizer cannot be loaded.
    """

    def __init__(self) -> None:
        self._cache: Dict[str, Any] = {}

    @staticmethod
    def resolve_tokenizer_id(model_name: str, override: Optional[str] = None) -> Optional[str]:
        """Maps an Ollama model tag (e.g., 'llama3.2:1b') to a Hugging Face repo ID or local path."""
        if override and override.strip():
            return override.strip()

        normalized = model_name.strip()
        if normalized in settings.tokenizer_map:
            return settings.tokenizer_map[normalized]

        lower_name = normalized.lower()
        for key, repo_id in settings.tokenizer_map.items():
            if lower_name.startswith(key.lower()):
                return repo_id

        # If the user passed a direct Hugging Face style 'org/repo' identifier
        if "/" in normalized and ":" not in normalized:
            return normalized

        return None

    def count_prompt_tokens(
        self,
        model_name: str,
        prompt: str,
        tokenizer_override: Optional[str] = None,
    ) -> TokenizerResult:
        """
        Tokenizes `prompt` using the exact model tokenizer.
        Never falls back to character-length heuristics.
        """
        if not prompt:
            return TokenizerResult(
                status=TokenizerStatus.ERROR,
                tokenizer_identifier=None,
                prompt_tokens=None,
                error="Prompt must be a non-empty string.",
            )

        if not TRANSFORMERS_AVAILABLE or AutoTokenizer is None:
            return TokenizerResult(
                status=TokenizerStatus.UNAVAILABLE,
                tokenizer_identifier=None,
                prompt_tokens=None,
                error="Hugging Face 'transformers' library is not installed; exact tokenization unavailable.",
            )

        tokenizer_id = self.resolve_tokenizer_id(model_name, tokenizer_override)
        if not tokenizer_id:
            return TokenizerResult(
                status=TokenizerStatus.UNAVAILABLE,
                tokenizer_identifier=None,
                prompt_tokens=None,
                error=(
                    f"No exact tokenizer mapping configured for model '{model_name}'. "
                    "Configure PREM_TOKENIZER_MAP or supply 'tokenizer_override'. "
                    "Character-count approximations are disabled by policy."
                ),
            )

        try:
            if tokenizer_id not in self._cache:
                logger.info(
                    "Loading Hugging Face tokenizer '%s' (local_files_only=%s)",
                    tokenizer_id,
                    settings.tokenizer_local_files_only,
                )
                self._cache[tokenizer_id] = AutoTokenizer.from_pretrained(
                    tokenizer_id,
                    token=settings.hf_token,
                    local_files_only=settings.tokenizer_local_files_only,
                    trust_remote_code=False,
                )

            tokenizer = self._cache[tokenizer_id]
            encoded = tokenizer.encode(prompt, add_special_tokens=True)
            token_count = len(encoded)

            logger.info(
                "Tokenized prompt %s with '%s' -> %d tokens",
                fingerprint_prompt(prompt),
                tokenizer_id,
                token_count,
            )

            return TokenizerResult(
                status=TokenizerStatus.EXACT_HF,
                tokenizer_identifier=tokenizer_id,
                prompt_tokens=token_count,
                error=None,
            )

        except Exception as exc:
            err_msg = (
                f"Failed to load or execute exact tokenizer '{tokenizer_id}' for model '{model_name}': {str(exc)}"
            )
            logger.warning(err_msg)
            return TokenizerResult(
                status=TokenizerStatus.UNAVAILABLE,
                tokenizer_identifier=tokenizer_id,
                prompt_tokens=None,
                error=err_msg,
            )


tokenizer_profiler = TokenizerProfiler()