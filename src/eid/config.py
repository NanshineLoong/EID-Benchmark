"""Configuration management for EID-Benchmark.

Handles model configuration, API keys, and environment settings.
Uses OPENAI_API_BASE_URL and OPENAI_API_KEY for all model access.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


@dataclass
class ModelConfig:
    """Configuration for a model instance.

    Attributes:
        model_name: Model identifier (e.g., gpt-5-mini)
        api_key: API key for authentication
        api_url: API endpoint URL
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts
        extra_config: Additional model-specific configuration
    """

    model_name: str
    api_key: str | None = None
    api_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 6000
    timeout: int = 120
    max_retries: int = 3
    extra_config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Load API credentials from environment if not provided."""
        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_url is None:
            self.api_url = os.getenv("OPENAI_API_BASE_URL")

    @classmethod
    def from_string(cls, model_name: str, **overrides: Any) -> "ModelConfig":
        """Create ModelConfig from a model name string.

        Args:
            model_name: Model name (e.g., gpt-5-mini)
            **overrides: Additional configuration overrides

        Returns:
            ModelConfig instance
        """
        config = cls(model_name=model_name)

        # Apply overrides
        for key, value in overrides.items():
            if hasattr(config, key) and value is not None:
                setattr(config, key, value)

        return config

    def to_camel_config(self) -> dict[str, Any]:
        """Convert to CAMEL-compatible configuration dictionary.

        Returns:
            Dictionary with CAMEL model configuration
        """
        config: dict[str, Any] = {
            "model_platform": "openai",
            "model_type": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }

        if self.api_key:
            config["api_key"] = self.api_key
        if self.api_url:
            config["url"] = self.api_url
        if self.extra_config:
            config["model_config_dict"] = self.extra_config

        return config


def load_config(env_path: str | Path | None = None) -> None:
    """Load environment configuration from .env file.

    Args:
        env_path: Path to .env file (optional, defaults to .env in cwd)
    """
    if env_path:
        load_dotenv(env_path)
    else:
        load_dotenv()


def get_model_config(
    model_name: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> ModelConfig:
    """Get model configuration from model name.

    Args:
        model_name: Model name (e.g., gpt-5-mini)
        temperature: Override temperature setting
        max_tokens: Override max_tokens setting
        **kwargs: Additional configuration overrides

    Returns:
        ModelConfig instance
    """
    # Get defaults from environment
    default_temp = float(os.getenv("DEFAULT_TEMPERATURE", "0.0"))
    default_max_tokens = int(os.getenv("DEFAULT_MAX_TOKENS", "6000"))

    return ModelConfig.from_string(
        model_name,
        temperature=temperature if temperature is not None else default_temp,
        max_tokens=max_tokens if max_tokens is not None else default_max_tokens,
        **kwargs,
    )


@dataclass
class EvaluationConfig:
    """Configuration for a complete evaluation run.

    Attributes:
        doctor_model: Model config for doctor/diagnostician role
        patient_model: Model config for patient simulator
        measurement_model: Model config for measurement/test simulator
        annotator_model: Model config for evaluation judge
        summarizer_model: Model config for summarizer role (SC/REFINE modes)
        diagnostician_model: Model config for diagnostician role (SC/REFINE modes)
        verifier_model: Model config for verifier role (REFINE mode)
        max_turns: Maximum interaction turns for roleplay modes
        max_workers: Maximum parallel workers
        output_dir: Directory for output files
    """

    doctor_model: ModelConfig
    patient_model: ModelConfig | None = None
    measurement_model: ModelConfig | None = None
    annotator_model: ModelConfig | None = None
    summarizer_model: ModelConfig | None = None
    diagnostician_model: ModelConfig | None = None
    verifier_model: ModelConfig | None = None
    max_turns: int = 16
    max_workers: int = 10
    output_dir: Path = field(default_factory=lambda: Path("results"))

    def __post_init__(self) -> None:
        """Set default models for roles if not specified."""
        default_model = os.getenv("DEFAULT_MODEL", "gpt-5-mini")

        if self.patient_model is None:
            self.patient_model = ModelConfig.from_string(default_model)
        if self.measurement_model is None:
            self.measurement_model = ModelConfig.from_string(default_model)
        if self.annotator_model is None:
            self.annotator_model = ModelConfig.from_string(default_model)
        if self.summarizer_model is None:
            self.summarizer_model = self.doctor_model
        if self.diagnostician_model is None:
            self.diagnostician_model = self.doctor_model
        if self.verifier_model is None:
            self.verifier_model = self.doctor_model
