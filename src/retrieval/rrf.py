"""
src/retrieval/rrf.py

RRF (Reciprocal Rank Fusion) retriever — combines BM25 and KNN run files.

Cormack et al. 2009: score(d) = 1/(k + rank_in_A) + 1/(k + rank_in_B)
k=60 is the standard constant that prevents very high ranks from dominating.

No new infrastructure needed — calls BM25Retriever and KNNRetriever, merges results.
The search() interface is identical to other retrievers so evaluation code is uniform.
"""

import os
import sys
from pathlib import Path

from opensearchpy import OpenSearch

from src.retrieval.base import BaseRetriever
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.knn import KNNRetriever
from src.embeddings.encoder import Encoder


# Merge two ranked result lists using Reciprocal Rank Fusion (k=60).
def rrf_merge(
    run_a: list[tuple[str, float]],
    run_b: list[tuple[str, float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Merge two ranked lists using RRF (Cormack 2009).

    Score: 1/(k + rank_in_a) + 1/(k + rank_in_b); 0 for docs absent from a list.
    Result is sorted by fused score descending.

    Args:
        run_a: ranked list of (doc_id, score) from first system.
        run_b: ranked list of (doc_id, score) from second system.
        k:     RRF constant (default 60 per Cormack 2009).

    Returns:
        Merged ranked list of (doc_id, fused_score), sorted descending.
    """
    scores: dict[str, float] = {}
    for rank, (pmid, _) in enumerate(run_a, start=1):
        scores[pmid] = scores.get(pmid, 0.0) + 1.0 / (k + rank)
    for rank, (pmid, _) in enumerate(run_b, start=1):
        scores[pmid] = scores.get(pmid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class RRFRetriever(BaseRetriever):
    """
    RRF fusion of BM25 and KNN results.

    Internally calls both retrievers and merges with rrf_merge().
    The final list is truncated to `size` after merging (union of both lists,
    then top-size by fused score).
    """

    def __init__(
        self,
        client: OpenSearch,
        index_name: str,
        encoder: Encoder | None = None,
        rrf_k: int = 60,
    ):
        """
        Args:
            encoder: shared Encoder instance to avoid loading the model twice.
            rrf_k:   RRF constant (default 60).
        """
        enc = encoder if encoder is not None else Encoder()
        self.bm25 = BM25Retriever(client, index_name)
        self.knn  = KNNRetriever(client, index_name, encoder=enc)
        self.rrf_k = rrf_k

    # Retrieve from BM25 + KNN independently, then fuse with RRF.
    def search(self, query: str, size: int = 100) -> list[tuple[str, float]]:
        # over-fetch slightly so fusion has more candidates to choose from
        fetch_size = size  # both lists have `size` results already
        bm25_results = self.bm25.search(query, size=fetch_size)
        knn_results  = self.knn.search(query, size=fetch_size)

        merged = rrf_merge(bm25_results, knn_results, k=self.rrf_k)
        return merged[:size]


# ---------------------------------------------------------------------------
# Self-test: python -m src.retrieval.rrf
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("RRF Retriever — self-test")
    print("=" * 60)

    import os
    from dotenv import load_dotenv
    from src.indexing.opensearch_client import get_client

    load_dotenv()
    client = get_client()
    index_name = os.getenv("OPENSEARCH_INDEX", "")

    print("\nLoading encoder ...")
    enc = Encoder()

    retriever = RRFRetriever(client, index_name, encoder=enc)

    test_query = "obstructive sleep apnea treatment"
    print(f"\nQuery: '{test_query}'")
    results = retriever.search(test_query, size=100)

    assert isinstance(results, list), "Result must be a list"
    assert len(results) == 100, f"Expected 100 results, got {len(results)}"
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True), "Scores not sorted descending"
    pmids = [p for p, _ in results]
    assert len(set(pmids)) == len(pmids), "Duplicate PMIDs in results"

    # RRF scores should be small positive floats (1/(60+1) + 1/(60+1) ≈ 0.032 max for rank-1 in both)
    assert all(0.0 < s < 1.0 for _, s in results), "RRF scores should be in (0, 1)"

    print(f"  Results count  : {len(results)}  OK")
    print(f"  Score order    : descending  OK")
    print(f"  No duplicates  : OK")
    print(f"  Score range    : ({scores[-1]:.6f}, {scores[0]:.6f})  OK")
    print(f"\n  Top 3 results:")
    for pmid, score in results[:3]:
        print(f"    PMID={pmid}  score={score:.6f}")

    # also test rrf_merge independently
    print("\n  rrf_merge unit test ...")
    list_a = [("A", 1.0), ("B", 0.9), ("C", 0.8)]
    list_b = [("B", 1.0), ("D", 0.9), ("A", 0.8)]
    merged = rrf_merge(list_a, list_b, k=60)
    merged_ids = [p for p, _ in merged]
    # B is rank 2 in A and rank 1 in B -- highest combined score
    assert merged_ids[0] == "B", f"B should rank first, got {merged_ids[0]}"
    # A is rank 1 in A and rank 3 in B -- second
    assert merged_ids[1] == "A", f"A should rank second, got {merged_ids[1]}"
    assert len(merged) == 4, f"Expected 4 unique PMIDs, got {len(merged)}"
    print("    rrf_merge unit test  OK")

    print("\n  All RRF assertions passed.")
