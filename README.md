# Qdrant Hybrid Search Engine

A production-ready hybrid search system that combines dense and sparse vectors with Reciprocal Rank Fusion (RRF), built for the [Qdrant Essentials Day 3](https://qdrant.tech/course/essentials/day-3/pitstop-project/) course project.

## What It Does

- **Dense vector search** for semantic understanding (sentence-transformers/all-MiniLM-L6-v2, 384-dim)
- **Sparse vector search** for exact keyword matching (term frequency with global vocabulary)
- **Hybrid search with RRF** combining results from both approaches via Qdrant's built-in Reciprocal Rank Fusion
- **DBSF fusion** as an alternative strategy (Distribution-Based Score Fusion)
- **Side-by-side comparison** between dense, sparse, and hybrid with overlap analysis
- **Latency benchmarking** across all search methods

## Sample Output

```
Query: 'how to understand text meaning with AI'

DENSE SEARCH:
  1. Retrieval Augmented Generation RAG (0.465)
  2. Natural Language Processing Fundamentals (0.361)
  3. Building Search Engines with Python (0.317)

SPARSE SEARCH:
  1. Retrieval Augmented Generation RAG (3.000)
  2. Building Search Engines with Python (2.000)
  3. Natural Language Processing Fundamentals (1.000)

HYBRID SEARCH (RRF):
  1. Retrieval Augmented Generation RAG (1.000)
  2. Building Search Engines with Python (0.583)
  3. Natural Language Processing Fundamentals (0.500)
  4. Embedding Models Comparison (0.343)
  5. Redis Caching Patterns (0.303) [HYBRID-ONLY]

Overlap Summary: {'shared_all': 3, 'unique_dense': 1, 'unique_sparse': 2, 'unique_hybrid': 1}
```

## Setup

### Prerequisites

- Python 3.10+
- A [Qdrant Cloud](https://cloud.qdrant.io/) cluster (free tier works)

### Installation

```bash
pip install -r requirements.txt
```

Or with dev dependencies:

```bash
pip install -e ".[dev]"
```

### Configuration

Copy the example env file and fill in your Qdrant credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```
QDRANT_URL=https://your-cluster-url.cloud.qdrant.io
QDRANT_API_KEY=your-api-key
COLLECTION_NAME=day3_hybrid_search
```

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `QDRANT_URL` | Yes | — | Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | Yes | — | Qdrant API authentication key |
| `DENSE_MODEL_NAME` | No | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace model identifier |
| `COLLECTION_NAME` | No | `day3_hybrid_search` | Qdrant collection name |
| `TOP_K` | No | `10` | Default number of results (1-100) |

## Usage

Run the full demonstration:

```bash
python main.py
```

This will:

1. Create a hybrid collection in Qdrant Cloud with dense (384-dim, cosine) and sparse vector configs
2. Ingest 20 sample documents (AI/ML/search domain) with both vector types
3. Run comparative searches showing dense vs sparse vs hybrid results
4. Compare RRF vs DBSF fusion strategies
5. Benchmark latency across all three approaches

## Project Structure

```
qdrant-hybrid-search-engine/
├── main.py                   # Entry point - full demonstration
├── src/
│   ├── __init__.py
│   ├── config.py             # Settings via environment variables
│   ├── exceptions.py         # Custom error hierarchy
│   ├── dense_encoder.py      # SentenceTransformer wrapper (all-MiniLM-L6-v2)
│   ├── sparse_encoder.py     # Term frequency with global vocabulary
│   ├── collection.py         # Collection setup + document ingestion
│   ├── search.py             # Dense, sparse, hybrid search with prefetch + FusionQuery
│   └── comparison.py         # Side-by-side comparison and benchmarking
├── requirements.txt
├── pyproject.toml
├── .env.example
└── README.md
```

## How It Works

### Sparse Encoding

Uses a global vocabulary that grows dynamically as new texts are processed. Each unique word gets a sequential integer index, and values are raw term frequency counts:

```python
from src.sparse_encoder import create_sparse_vector

vector = create_sparse_vector("hello world hello")
# indices: [0, 1]  (hello=0, world=1)
# values: [2.0, 1.0]  (hello appears 2x, world 1x)
```

### Hybrid Search with RRF

Uses Qdrant's built-in fusion via `query_points` with `prefetch`:

```python
response = client.query_points(
    collection_name="day3_hybrid_search",
    prefetch=[
        models.Prefetch(query=dense_vector, using="dense", limit=20),
        models.Prefetch(query=sparse_vector, using="sparse", limit=20),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    limit=10,
)
```

### DBSF Fusion

Distribution-Based Score Fusion as an alternative — just swap `Fusion.RRF` for `Fusion.DBSF`:

```python
query=models.FusionQuery(fusion=models.Fusion.DBSF)
```

## Performance Results

Tested with 5 queries, 5 iterations each against Qdrant Cloud:

| Approach | Avg Latency | Min | Max |
|----------|-------------|-----|-----|
| Dense    | ~197 ms     | ~186 ms | ~212 ms |
| Sparse   | ~175 ms     | ~171 ms | ~185 ms |
| Hybrid   | ~198 ms     | ~185 ms | ~209 ms |

Sparse is fastest (no model encoding on client), hybrid adds minimal overhead over dense since fusion runs server-side.

## Key Findings

- **Hybrid beats single-vector** for mixed queries (keywords + concepts) — it surfaces documents that neither approach finds alone
- **RRF vs DBSF** produce different rankings; DBSF tends to give higher absolute scores and sometimes reorders results differently
- **Sparse excels** at exact keyword matching (e.g., "BM25 scoring term frequency")
- **Dense excels** at semantic paraphrases (e.g., "how to understand text meaning" finds NLP/RAG docs)
- **Hybrid-only results** appear when a document ranks moderately in both lists but top in neither

## Dependencies

- `qdrant-client >= 1.9.0` — Qdrant Python SDK
- `sentence-transformers >= 2.2.0` — Dense embedding model
- `numpy >= 1.24.0` — Numerical operations
- `python-dotenv >= 1.0.0` — Environment variable management

## License

MIT
