"""Qdrant collection management and document ingestion.

Handles creating hybrid collections with both dense and sparse vectors,
and uploading documents with their encoded representations.
"""

import logging

from qdrant_client import QdrantClient, models

from src.config import Settings
from src.dense_encoder import DenseEncoder
from src.exceptions import CollectionError
from src.sparse_encoder import create_sparse_vector

logger = logging.getLogger(__name__)

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


def setup_collection(client: QdrantClient, settings: Settings) -> dict:
    """Create a hybrid collection with dense and sparse vector configurations.

    If the collection already exists, it is deleted and recreated to ensure
    a clean state. Follows the Qdrant Essentials Day 3 pattern.

    Args:
        client: Initialized QdrantClient.
        settings: Application settings with collection_name.

    Returns:
        Dict with collection_name, vector_configs, and point_count.

    Raises:
        CollectionError: If collection creation or verification fails.
    """
    collection_name = settings.collection_name

    try:
        # Delete existing collection if present
        collections = client.get_collections().collections
        existing_names = [c.name for c in collections]
        if collection_name in existing_names:
            client.delete_collection(collection_name=collection_name)
            logger.info(f"Deleted existing collection '{collection_name}'")

        # Create hybrid collection (same as course reference)
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=384, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                )
            },
        )

        # Verify collection exists
        info = client.get_collection(collection_name=collection_name)
        result = {
            "collection_name": collection_name,
            "vector_configs": {
                "dense": {"size": 384, "distance": "Cosine"},
                "sparse": {"name": SPARSE_VECTOR_NAME},
            },
            "point_count": info.points_count or 0,
        }
        logger.info(
            f"Created collection '{collection_name}' with dense (384, cosine) "
            f"and sparse vectors. Points: {result['point_count']}"
        )
        return result

    except CollectionError:
        raise
    except Exception as e:
        raise CollectionError(
            f"Failed to create collection '{collection_name}': {e}"
        ) from e


def ingest_documents(
    client: QdrantClient,
    settings: Settings,
    documents: list[dict],
    dense_encoder: DenseEncoder,
) -> dict:
    """Encode and upload documents with both dense and sparse vectors.

    Each document must have 'title' and 'description' fields. The text used
    for encoding is the description (matching the course reference).
    Documents are uploaded using PointStruct with vector and sparse_vector
    fields.

    Args:
        client: Initialized QdrantClient.
        settings: Application settings.
        documents: List of document dicts with title, description, metadata.
        dense_encoder: Initialized DenseEncoder for dense embeddings.

    Returns:
        Dict with total_indexed and total_failed counts.
    """
    collection_name = settings.collection_name
    batch_size = settings.batch_size
    total_indexed = 0
    total_failed = 0

    # Build points - sparse vectors go inside the vector dict (qdrant-client >= 1.12)
    points = []
    for i, item in enumerate(documents):
        try:
            description = item.get("description", "")
            dense_vector = dense_encoder.encode(description)
            sparse_vector = create_sparse_vector(description)

            points.append(
                models.PointStruct(
                    id=i,
                    vector={
                        DENSE_VECTOR_NAME: dense_vector,
                        SPARSE_VECTOR_NAME: sparse_vector,
                    },
                    payload=item,
                )
            )
            total_indexed += 1
        except Exception as e:
            logger.warning(f"Failed to process document {i}: {e}")
            total_failed += 1

    # Upload in batches
    for batch_start in range(0, len(points), batch_size):
        batch = points[batch_start : batch_start + batch_size]
        try:
            client.upload_points(
                collection_name=collection_name, points=batch
            )
        except Exception as e:
            logger.warning(f"Failed to upload batch starting at {batch_start}: {e}")
            total_failed += len(batch)
            total_indexed -= len(batch)

    logger.info(
        f"Ingestion complete: {total_indexed} indexed, {total_failed} failed"
    )
    return {"total_indexed": total_indexed, "total_failed": total_failed}
