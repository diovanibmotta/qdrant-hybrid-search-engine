"""Search comparison and performance benchmarking.

Provides side-by-side comparison of dense, sparse, and hybrid search results,
along with latency benchmarking across all approaches.
"""

import time
from dataclasses import dataclass, field

from src.search import FusionMethod, SearchOperations, SearchResult


@dataclass
class ComparisonResult:
    """Results from comparing all three search approaches."""

    query: str
    dense_results: list[SearchResult]
    sparse_results: list[SearchResult]
    hybrid_results: list[SearchResult]
    overlap_summary: dict = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Latency benchmark results for a single search approach."""

    approach: str
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    total_queries: int
    failed_queries: int


class Comparison:
    """Executes search comparisons and performance benchmarks.

    Args:
        search_ops: Initialized SearchOperations instance.
    """

    def __init__(self, search_ops: SearchOperations):
        self.search_ops = search_ops

    def compare(self, query: str, limit: int | None = None) -> ComparisonResult:
        """Execute dense, sparse, and hybrid searches for the same query.

        Computes overlap summary showing shared and unique documents
        across all three approaches.

        Args:
            query: Search query text.
            limit: Max results per approach.

        Returns:
            ComparisonResult with all three result sets and overlap summary.
        """
        dense_results = self.search_ops.dense_search(query, limit)
        sparse_results = self.search_ops.sparse_search(query, limit)
        hybrid_results = self.search_ops.hybrid_search(query, limit)

        # Compute overlap summary
        dense_titles = {r.title for r in dense_results}
        sparse_titles = {r.title for r in sparse_results}
        hybrid_titles = {r.title for r in hybrid_results}

        shared_all = dense_titles & sparse_titles & hybrid_titles
        unique_dense = dense_titles - sparse_titles - hybrid_titles
        unique_sparse = sparse_titles - dense_titles - hybrid_titles
        unique_hybrid = hybrid_titles - dense_titles - sparse_titles

        overlap_summary = {
            "shared_all": len(shared_all),
            "unique_dense": len(unique_dense),
            "unique_sparse": len(unique_sparse),
            "unique_hybrid": len(unique_hybrid),
        }

        return ComparisonResult(
            query=query,
            dense_results=dense_results,
            sparse_results=sparse_results,
            hybrid_results=hybrid_results,
            overlap_summary=overlap_summary,
        )

    def benchmark(
        self, queries: list[str], limit: int | None = None, iterations: int = 10
    ) -> list[BenchmarkResult]:
        """Benchmark all three search approaches across a set of queries.

        Measures end-to-end latency including encoding and retrieval.
        Follows the course reference benchmarking pattern.

        Args:
            queries: List of query texts (recommend at least 5).
            limit: Max results per search.
            iterations: Number of iterations per query per method.

        Returns:
            List of BenchmarkResult, one per approach.
        """
        methods = {
            "dense": lambda q: self.search_ops.dense_search(q, limit),
            "sparse": lambda q: self.search_ops.sparse_search(q, limit),
            "hybrid": lambda q: self.search_ops.hybrid_search(q, limit),
        }

        results = []

        for method_name, method_func in methods.items():
            times = []
            failed = 0

            for query in queries:
                for _ in range(iterations):
                    try:
                        start = time.time()
                        method_func(query)
                        elapsed_ms = (time.time() - start) * 1000
                        times.append(elapsed_ms)
                    except Exception:
                        failed += 1

            if times:
                results.append(
                    BenchmarkResult(
                        approach=method_name,
                        avg_latency_ms=sum(times) / len(times),
                        min_latency_ms=min(times),
                        max_latency_ms=max(times),
                        total_queries=len(queries) * iterations,
                        failed_queries=failed,
                    )
                )
            else:
                results.append(
                    BenchmarkResult(
                        approach=method_name,
                        avg_latency_ms=0.0,
                        min_latency_ms=0.0,
                        max_latency_ms=0.0,
                        total_queries=len(queries) * iterations,
                        failed_queries=failed,
                    )
                )

        return results

    def display_comparison(self, result: ComparisonResult) -> None:
        """Print side-by-side comparison of search results."""
        print(f"Query: '{result.query}'\n")

        print("DENSE SEARCH:")
        for r in result.dense_results:
            print(f"  {r.rank}. {r.title} ({r.score:.3f})")

        print("\nSPARSE SEARCH:")
        for r in result.sparse_results:
            print(f"  {r.rank}. {r.title} ({r.score:.3f})")

        print("\nHYBRID SEARCH (RRF):")
        # Annotate hybrid-only results
        dense_titles = {r.title for r in result.dense_results}
        sparse_titles = {r.title for r in result.sparse_results}
        for r in result.hybrid_results:
            marker = ""
            if r.title not in dense_titles and r.title not in sparse_titles:
                marker = " [HYBRID-ONLY]"
            print(f"  {r.rank}. {r.title} ({r.score:.3f}){marker}")

        print(f"\nOverlap Summary: {result.overlap_summary}")
        print("-" * 50)

    def display_benchmark(self, results: list[BenchmarkResult]) -> None:
        """Print benchmark summary table."""
        print("\nPerformance Benchmark:")
        print(f"{'Approach':<10} {'Avg (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12} {'Failed'}")
        print("-" * 60)
        for r in results:
            print(
                f"{r.approach:<10} {r.avg_latency_ms:<12.2f} "
                f"{r.min_latency_ms:<12.2f} {r.max_latency_ms:<12.2f} "
                f"{r.failed_queries}"
            )
