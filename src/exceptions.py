"""Custom exception hierarchy for the Hybrid Search Engine."""


class HybridSearchError(Exception):
    """Base exception for all hybrid search engine errors."""

    pass


class ConfigurationError(HybridSearchError):
    """Raised for missing or invalid configuration.

    Examples:
        - Missing required environment variable: {name}
        - Invalid top-k value '{value}': must be integer in range [1, 100]
    """

    pass


class ConnectionError(HybridSearchError):
    """Raised when Qdrant cluster is unreachable.

    Examples:
        - Cannot connect to Qdrant at {url}: {reason}
    """

    pass


class AuthenticationError(HybridSearchError):
    """Raised when API key is invalid.

    Examples:
        - Authentication failed for Qdrant: API key rejected
    """

    pass


class CollectionError(HybridSearchError):
    """Raised for collection creation/verification failures.

    Examples:
        - Failed to create collection '{name}': {reason}
    """

    pass


class EncodingError(HybridSearchError):
    """Raised when dense or sparse encoding fails.

    Examples:
        - Failed to load model '{model_name}': {reason}
        - Encoding failed for query '{text[:50]}': {reason}
    """

    pass
