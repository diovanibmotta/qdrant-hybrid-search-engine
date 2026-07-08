"""Hybrid Search Engine - Main entry point.

Demonstrates the complete hybrid search pipeline:
1. Setup hybrid collection with dense and sparse vectors
2. Ingest sample documents
3. Run dense, sparse, and hybrid searches
4. Compare results side-by-side
5. Benchmark performance

Based on Qdrant Essentials Day 3: Building a Hybrid Search Engine.
"""

import logging
import sys

from qdrant_client import QdrantClient

from src.collection import ingest_documents, setup_collection
from src.comparison import Comparison
from src.config import load_settings
from src.dense_encoder import DenseEncoder
from src.exceptions import HybridSearchError
from src.search import FusionMethod, SearchOperations
from src.sparse_encoder import reset_vocabulary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Sample dataset - Technology & AI domain
SAMPLE_DOCUMENTS = [
    {
        "title": "Introduction to Machine Learning",
        "description": "An overview of supervised and unsupervised learning techniques including decision trees, random forests, and clustering algorithms for beginners.",
        "metadata": {"category": "ai", "level": "beginner"},
    },
    {
        "title": "Deep Learning with Neural Networks",
        "description": "Exploring convolutional neural networks CNNs and recurrent neural networks RNNs for image classification and natural language processing tasks.",
        "metadata": {"category": "ai", "level": "advanced"},
    },
    {
        "title": "Natural Language Processing Fundamentals",
        "description": "Tokenization word embeddings and transformer models like BERT and GPT for understanding human language and text classification.",
        "metadata": {"category": "nlp", "level": "intermediate"},
    },
    {
        "title": "Vector Databases for Similarity Search",
        "description": "How vector databases like Qdrant enable fast approximate nearest neighbor search at scale using HNSW indexing and quantization techniques.",
        "metadata": {"category": "databases", "level": "intermediate"},
    },
    {
        "title": "Building Search Engines with Python",
        "description": "A practical guide to implementing full-text search and semantic search using Python libraries including Whoosh and sentence-transformers.",
        "metadata": {"category": "search", "level": "beginner"},
    },
    {
        "title": "Retrieval Augmented Generation RAG",
        "description": "Combining vector search with large language models to ground AI responses in factual retrieved documents reducing hallucination.",
        "metadata": {"category": "ai", "level": "advanced"},
    },
    {
        "title": "Kubernetes Container Orchestration",
        "description": "Deploying and managing containerized applications at scale using Kubernetes pods services and deployments with auto-scaling.",
        "metadata": {"category": "devops", "level": "intermediate"},
    },
    {
        "title": "Python Data Science Libraries",
        "description": "Using pandas numpy and scikit-learn for data analysis feature engineering and building predictive machine learning models.",
        "metadata": {"category": "data-science", "level": "beginner"},
    },
    {
        "title": "Microservices Architecture Patterns",
        "description": "Designing distributed systems with event-driven architecture API gateways service mesh and circuit breaker patterns for resilience.",
        "metadata": {"category": "architecture", "level": "advanced"},
    },
    {
        "title": "Graph Neural Networks",
        "description": "Applying deep learning to graph-structured data for social network analysis molecular property prediction and recommendation systems.",
        "metadata": {"category": "ai", "level": "advanced"},
    },
    {
        "title": "FastAPI Web Framework",
        "description": "Building high-performance REST APIs with Python FastAPI including automatic OpenAPI documentation async support and dependency injection.",
        "metadata": {"category": "web", "level": "intermediate"},
    },
    {
        "title": "Embedding Models Comparison",
        "description": "Comparing sentence-transformers models like all-MiniLM-L6-v2 BGE and E5 for semantic similarity text retrieval and clustering tasks.",
        "metadata": {"category": "nlp", "level": "intermediate"},
    },
    {
        "title": "Database Indexing Strategies",
        "description": "B-tree hash and inverted index structures for optimizing query performance in relational and NoSQL databases.",
        "metadata": {"category": "databases", "level": "intermediate"},
    },
    {
        "title": "Sparse Retrieval with BM25",
        "description": "Understanding BM25 scoring term frequency inverse document frequency and keyword-based information retrieval for search engines.",
        "metadata": {"category": "search", "level": "intermediate"},
    },
    {
        "title": "Hybrid Search Fusion Techniques",
        "description": "Reciprocal rank fusion and distribution-based score fusion methods for combining dense semantic and sparse keyword search results.",
        "metadata": {"category": "search", "level": "advanced"},
    },
    {
        "title": "Docker Containerization Basics",
        "description": "Creating Dockerfiles building images and running containers for reproducible development environments and CI/CD pipelines.",
        "metadata": {"category": "devops", "level": "beginner"},
    },
    {
        "title": "Attention Mechanism in Transformers",
        "description": "Self-attention multi-head attention and positional encoding in transformer architecture used by BERT GPT and vision transformers.",
        "metadata": {"category": "ai", "level": "advanced"},
    },
    {
        "title": "Redis Caching Patterns",
        "description": "Implementing cache-aside write-through and pub/sub patterns with Redis for improving application response times and throughput.",
        "metadata": {"category": "databases", "level": "intermediate"},
    },
    {
        "title": "Prompt Engineering for LLMs",
        "description": "Techniques for crafting effective prompts including chain-of-thought few-shot learning and system prompts for ChatGPT and Claude.",
        "metadata": {"category": "ai", "level": "beginner"},
    },
    {
        "title": "Approximate Nearest Neighbor Algorithms",
        "description": "HNSW LSH and IVF algorithms for efficient similarity search in high-dimensional vector spaces with trade-offs between recall and speed.",
        "metadata": {"category": "search", "level": "advanced"},
    },
]


def main():
    """Run the complete hybrid search demonstration."""
    print("=" * 60)
    print("  Hybrid Search Engine - Qdrant Essentials Day 3")
    print("=" * 60)
    print()

    # Step 1: Load configuration
    try:
        settings = load_settings()
        print(f"✓ Configuration loaded (collection: {settings.collection_name})")
    except HybridSearchError as e:
        print(f"✗ Configuration error: {e}")
        sys.exit(1)

    # Step 2: Initialize client and encoders
    try:
        client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=10,
        )
        print("✓ Qdrant client initialized")

        dense_encoder = DenseEncoder(settings.dense_model_name)
        print(f"✓ Dense encoder loaded ({settings.dense_model_name})")

        # Reset sparse vocabulary for clean start
        reset_vocabulary()
        print("✓ Sparse encoder ready (global vocabulary)")

    except HybridSearchError as e:
        print(f"✗ Initialization error: {e}")
        sys.exit(1)

    # Step 3: Setup collection and ingest documents
    try:
        collection_info = setup_collection(client, settings)
        print(
            f"✓ Collection '{collection_info['collection_name']}' created "
            f"(dense: 384-dim cosine, sparse: keyword)"
        )

        result = ingest_documents(client, settings, SAMPLE_DOCUMENTS, dense_encoder)
        print(
            f"✓ Ingested {result['total_indexed']} documents "
            f"({result['total_failed']} failed)"
        )
    except HybridSearchError as e:
        print(f"✗ Collection/ingestion error: {e}")
        sys.exit(1)

    print()

    # Step 4: Initialize search operations
    search_ops = SearchOperations(client, settings, dense_encoder)
    comparison = Comparison(search_ops)

    # Step 5: Demonstrate searches
    print("=" * 60)
    print("  Search Demonstrations")
    print("=" * 60)
    print()

    # Test queries that show different strengths
    test_queries = [
        # Semantic query - dense should excel
        "how to understand text meaning with AI",
        # Keyword query - sparse should excel
        "BM25 scoring term frequency",
        # Mixed query - hybrid should combine strengths
        "vector database similarity search HNSW",
    ]

    for query in test_queries:
        result = comparison.compare(query, limit=5)
        comparison.display_comparison(result)
        print()

    # Step 6: RRF vs DBSF comparison
    print("=" * 60)
    print("  RRF vs DBSF Fusion Comparison")
    print("=" * 60)
    print()

    demo_query = "semantic search with embeddings"
    print(f"Query: '{demo_query}'\n")

    rrf_results = search_ops.hybrid_search(demo_query, limit=5, fusion=FusionMethod.RRF)
    dbsf_results = search_ops.hybrid_search(
        demo_query, limit=5, fusion=FusionMethod.DBSF
    )

    print("RRF Results:")
    for r in rrf_results:
        print(f"  {r.rank}. {r.title} ({r.score:.3f})")

    print("\nDBSF Results:")
    for r in dbsf_results:
        print(f"  {r.rank}. {r.title} ({r.score:.3f})")

    print()

    # Step 7: Performance benchmark
    print("=" * 60)
    print("  Performance Benchmark")
    print("=" * 60)
    print()

    benchmark_queries = [
        "machine learning algorithms",
        "container orchestration kubernetes",
        "natural language understanding",
        "database indexing performance",
        "building REST APIs python",
    ]

    benchmark_results = comparison.benchmark(benchmark_queries, limit=5, iterations=5)
    comparison.display_benchmark(benchmark_results)

    print()
    print("=" * 60)
    print("  Done! Hybrid search engine demonstration complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
