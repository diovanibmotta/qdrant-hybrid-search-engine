"""Configuration management for the Hybrid Search Engine.

Reads settings from environment variables with support for .env files via python-dotenv.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from src.exceptions import ConfigurationError


@dataclass(frozen=True)
class Settings:
    """Immutable application settings loaded from environment variables."""

    qdrant_url: str
    qdrant_api_key: str
    dense_model_name: str
    collection_name: str
    top_k: int
    batch_size: int


def load_settings() -> Settings:
    """Load and validate settings from environment variables.

    Optionally loads a .env file if present. Reads required and optional
    environment variables, applies defaults, and validates constraints.

    Returns:
        Settings: A frozen dataclass with all configuration values.

    Raises:
        ConfigurationError: If QDRANT_URL or QDRANT_API_KEY is missing,
            or if TOP_K is not an integer in the range [1, 100].
    """
    load_dotenv()

    qdrant_url = os.environ.get("QDRANT_URL")
    if not qdrant_url:
        raise ConfigurationError("Missing required environment variable: QDRANT_URL")

    qdrant_api_key = os.environ.get("QDRANT_API_KEY")
    if not qdrant_api_key:
        raise ConfigurationError("Missing required environment variable: QDRANT_API_KEY")

    dense_model_name = os.environ.get(
        "DENSE_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
    )
    collection_name = os.environ.get("COLLECTION_NAME", "hybrid_collection")

    top_k_raw = os.environ.get("TOP_K", "10")
    try:
        top_k = int(top_k_raw)
    except (ValueError, TypeError):
        raise ConfigurationError(
            f"Invalid top-k value '{top_k_raw}': must be integer in range [1, 100]"
        )

    if top_k < 1 or top_k > 100:
        raise ConfigurationError(
            f"Invalid top-k value '{top_k}': must be integer in range [1, 100]"
        )

    return Settings(
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        dense_model_name=dense_model_name,
        collection_name=collection_name,
        top_k=top_k,
        batch_size=100,
    )
