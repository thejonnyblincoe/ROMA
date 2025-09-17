"""
Model Factory for ROMA v2.0

Creates LLM models using Agno's LiteLLM integration for unified access
to all providers through a single interface.
"""

from typing import Dict, Any, Optional
import os
import logging

from src.roma.domain.value_objects.config.model_config import ModelConfig

# Import with error handling
try:
    from agno.models.litellm import LiteLLM
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    LiteLLM = None

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory for creating LLM models using Agno's LiteLLM."""

    def __init__(self):
        """Initialize model factory with cache."""
        self._model_cache: Dict[str, Any] = {}
        self._provider_env_keys = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
            "gemini": "GOOGLE_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "together_ai": "TOGETHER_API_KEY",
            "groq": "GROQ_API_KEY",
            "cohere": "COHERE_API_KEY",
            "bedrock": "AWS_ACCESS_KEY_ID",
            "replicate": "REPLICATE_API_TOKEN",
            "huggingface": "HUGGINGFACE_API_TOKEN",
            "ollama": None  # Ollama typically doesn't need an API key
        }

    def create_model(self, config: ModelConfig):
        """
        Create an Agno-compatible model from configuration.

        Args:
            config: ModelConfig with provider and model details

        Returns:
            Agno LiteLLM model instance

        Raises:
            RuntimeError: If LiteLLM is not available
            ValueError: If configuration is invalid
        """
        if not LITELLM_AVAILABLE:
            raise RuntimeError("LiteLLM is required but not available. Install with: pip install litellm")

        cache_key = self._get_cache_key(config)

        if cache_key in self._model_cache:
            logger.debug(f"Using cached model: {cache_key}")
            return self._model_cache[cache_key]

        model = self._create_litellm_model(config)
        self._model_cache[cache_key] = model
        logger.info(f"Created LiteLLM model: {config.model_id}")
        return model

    def _create_litellm_model(self, config: ModelConfig) -> LiteLLM:
        """Create LiteLLM model for any provider."""
        # Build parameters - only include what's provided
        params = {"id": config.model_id}

        if config.temperature is not None:
            params["temperature"] = config.temperature
        if config.max_tokens is not None:
            params["max_tokens"] = config.max_tokens
        if config.top_p is not None:
            params["top_p"] = config.top_p

        # Handle API key
        if config.api_key:
            params["api_key"] = config.api_key
        else:
            # Try to get API key based on model prefix
            api_key = self._get_api_key_for_model(config.model_id)
            if api_key:
                params["api_key"] = api_key

        # Handle API base for custom endpoints (like Ollama)
        if config.api_base:
            params["api_base"] = config.api_base

        logger.debug(f"Creating LiteLLM model with params: {self._sanitize_params_for_log(params)}")
        return LiteLLM(**params)

    def _get_api_key_for_model(self, model_id: str) -> Optional[str]:
        """Get API key from environment based on model prefix."""
        # Extract provider from model_id prefix
        for provider, env_key in self._provider_env_keys.items():
            if model_id.startswith(f"{provider}/") or (provider == "openai" and "/" not in model_id):
                if env_key:
                    api_key = os.getenv(env_key)
                    if api_key:
                        logger.debug(f"Found API key for provider: {provider}")
                        return api_key
                return None  # No API key needed (e.g., Ollama)

        logger.debug(f"No API key mapping found for model: {model_id}")
        return None

    def _get_cache_key(self, config: ModelConfig) -> str:
        """Generate cache key for model configuration."""
        return f"{config.provider}:{config.model_id}:{config.temperature}:{config.max_tokens}"

    def _sanitize_params_for_log(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize parameters for logging (hide API keys)."""
        sanitized = params.copy()
        if "api_key" in sanitized and sanitized["api_key"]:
            sanitized["api_key"] = f"{sanitized['api_key'][:8]}..."
        return sanitized

    def clear_cache(self) -> None:
        """Clear the model cache."""
        self._model_cache.clear()
        logger.info("Model cache cleared")

    def get_cached_model_count(self) -> int:
        """Get number of cached models."""
        return len(self._model_cache)

    def is_model_cached(self, config: ModelConfig) -> bool:
        """Check if a model configuration is cached."""
        cache_key = self._get_cache_key(config)
        return cache_key in self._model_cache