"""Dense vector encoding using sentence-transformers.

Wraps the SentenceTransformer model for producing 384-dimensional dense
embeddings used in semantic search.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from src.exceptions import EncodingError


class DenseEncoder:
    """Encodes text into dense vector embeddings using sentence-transformers.

    Args:
        model_name: HuggingFace model identifier. Defaults to "all-MiniLM-L6-v2".
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Load the sentence-transformers model.

        Args:
            model_name: HuggingFace model identifier.

        Raises:
            EncodingError: If model fails to load.
        """
        try:
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
        except Exception as e:
            raise EncodingError(
                f"Failed to load model '{model_name}': {e}"
            ) from e

    def encode(self, text: str) -> list[float]:
        """Encode text into a dense vector (list of floats).

        Args:
            text: Input text to encode.

        Returns:
            List of floats representing the 384-dimensional embedding.

        Raises:
            ValueError: If text is empty or whitespace-only.
            EncodingError: If encoding fails.
        """
        if not text or not text.strip():
            raise ValueError("Query text must be non-empty")

        try:
            embedding = self.model.encode(text)
            return embedding.tolist()
        except Exception as e:
            truncated = text[:50]
            raise EncodingError(
                f"Encoding failed for query '{truncated}': {e}"
            ) from e

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode multiple texts efficiently using batch inference.

        Args:
            texts: List of input texts to encode.

        Returns:
            List of dense vectors (each a list of floats).
        """
        embeddings = self.model.encode(texts)
        return [emb.tolist() for emb in embeddings]
