"""Search operations: dense, sparse, and hybrid with RRF/DBSF fusion.

Implements all search approaches following the Qdrant Essentials Day 3 pattern,
using query_points with prefetch and FusionQuery for hybrid search.
"""

import logging
from dataclasses import dataclass
from enum import Enum

from qdrant_client import QdrantClient, models

from src.collection import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from src.config import Settings
from src.dense_encoder import DenseEncoder
from src.sparse_encoder import create_sparse_vector

logger = logging.getLogger(__name__)

_NO_TITLE = "No title"
_NO_MATCHES_WARNING = "No matches found for query"


class FusionMethod(Enum):
    """Available fusion methods for hybrid search."""

    RRF = "rrf"
    DBSF = "dbsf"


@dataclass
class SearchResult:
    """A single search result with rank, score, and document fields."""

    rank: int
    score: float
    title: str
    description: str


class SearchOperations:
    """Performs dense, sparse, and hybrid search operations on a Qdrant collection.

    Args:
        client: Initialized QdrantClient.
        settings: Application settings.
        dense_encoder: Initialized DenseEncoder.
    """

    def __init__(
        self,
        client: QdrantClient,
        settings: Settings,
        dense_encoder: DenseEncoder,
    ):
        self.client = client
        self.settings = settings
        self.dense_encoder = dense_encoder

    def _validate_query(self, query: str) -> None:
        """Validate query text is non-empty."""
        if not query or not query.strip():
            raise ValueError("Query text must be non-empty")

    def dense_search(self, query: str, limit: int | None = None) -> list[SearchResult]:
        """Perform dense-only semantic search.

        Args:
            query: Search query text.
            limit: Max results to return. Defaults to settings.top_k.

        Returns:
            List of SearchResult ordered by cosine similarity descending.

        Raises:
            ValueError: If query is empty/whitespace.
        """
        self._validate_query(query)
        limit = limit or self.settings.top_k

        query_dense = self.dense_encoder.encode(query)

        response = self.client.query_points(
            collection_name=self.settings.collection_name,
            query=query_dense,
            using=DENSE_VECTOR_NAME,
            limit=limit,
        )

        results = []
        for i, point in enumerate(response.points, 1):
            results.append(
                SearchResult(
                    rank=i,
                    score=point.score,
                    title=point.payload.get("title", _NO_TITLE),
                    description=point.payload.get("description", ""),
                )
            )

        if not results:
            logger.warning(_NO_MATCHES_WARNING)

        return results

    def sparse_search(self, query: str, limit: int | None = None) -> list[SearchResult]:
        """Perform sparse-only keyword search.

        Args:
            query: Search query text.
            limit: Max results to return. Defaults to settings.top_k.

        Returns:
            List of SearchResult ordered by sparse similarity descending.

        Raises:
            ValueError: If query is empty/whitespace.
        """
        self._validate_query(query)
        limit = limit or self.settings.top_k

        query_sparse = create_sparse_vector(query)

        response = self.client.query_points(
            collection_name=self.settings.collection_name,
            query=query_sparse,
            using=SPARSE_VECTOR_NAME,
            limit=limit,
        )

        results = []
        for i, point in enumerate(response.points, 1):
            results.append(
                SearchResult(
                    rank=i,
                    score=point.score,
                    title=point.payload.get("title", _NO_TITLE),
                    description=point.payload.get("description", ""),
                )
            )

        if not results:
            logger.warning(_NO_MATCHES_WARNING)

        return results

    def hybrid_search(
        self,
        query: str,
        limit: int | None = None,
        fusion: FusionMethod = FusionMethod.RRF,
    ) -> list[SearchResult]:
        """Perform hybrid search using Qdrant's built-in fusion (RRF or DBSF).

        Uses query_points with prefetch for both dense and sparse queries,
        then applies the specified fusion method server-side.

        Args:
            query: Search query text.
            limit: Max results to return. Defaults to settings.top_k.
            fusion: Fusion method to use (RRF or DBSF).

        Returns:
            List of SearchResult ordered by fused score descending.

        Raises:
            ValueError: If query is empty/whitespace.
        """
        self._validate_query(query)
        limit = limit or self.settings.top_k

        # Encode query for both dense and sparse
        query_dense = self.dense_encoder.encode(query)
        query_sparse = create_sparse_vector(query)

        # Select fusion method
        if fusion == FusionMethod.DBSF:
            qdrant_fusion = models.Fusion.DBSF
        else:
            qdrant_fusion = models.Fusion.RRF

        # Use Qdrant's built-in fusion (same as course reference)
        response = self.client.query_points(
            collection_name=self.settings.collection_name,
            prefetch=[
                models.Prefetch(
                    query=query_dense,
                    using=DENSE_VECTOR_NAME,
                    limit=20,
                ),
                models.Prefetch(
                    query=query_sparse,
                    using=SPARSE_VECTOR_NAME,
                    limit=20,
                ),
            ],
            query=models.FusionQuery(fusion=qdrant_fusion),
            limit=limit,
        )

        results = []
        for i, point in enumerate(response.points, 1):
            results.append(
                SearchResult(
                    rank=i,
                    score=point.score,
                    title=point.payload.get("title", _NO_TITLE),
                    description=point.payload.get("description", ""),
                )
            )

        if not results:
            logger.warning(_NO_MATCHES_WARNING)

        return results
