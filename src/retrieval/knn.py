"""
src/retrieval/knn.py

Dense KNN retriever -- encode query, then knn search on the `embedding` field.

Follows Lab01 KNN query pattern (lines 541-567): both "size" and "k" must be set.
Encoder uses the same model as corpus encoding so query and document vectors
are in the same space.
"""

import os
import sys
from pathlib import Path

import numpy as np
from opensearchpy import OpenSearch

from src.embeddings.encoder import Encoder
from src.retrieval.base import BaseRetriever


class KNNRetriever(BaseRetriever):
    """Dense KNN retrieval using msmarco-distilbert-base-v2 embeddings."""

    def __init__(self, client: OpenSearch, index_name: str, encoder: Encoder | None = None):
        """
        Args:
            encoder: an Encoder instance. If None, creates one (slow — pass one in for reuse).
        """
        self.client = client
        self.index_name = index_name
        # load encoder lazily only if not provided
        self.encoder = encoder if encoder is not None else Encoder()

    # Encode query, then run KNN search; size and k must both be set per Lab01 pattern.
    def search(self, query: str, size: int = 100) -> list[tuple[str, float]]:
        # encode the query using the same model+pooling as the indexed vectors
        query_vector = self.encoder.encode_single(query)

        query_body = {
            "size": size,
            "_source": ["doc_id"],
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_vector.tolist(),
                        "k": size,   # k = final result count from coordinator (not per-shard)
                    }
                }
            },
        }
        response = self.client.search(body=query_body, index=self.index_name)
        hits = response["hits"]["hits"]
        return [(h["_source"]["doc_id"], h["_score"]) for h in hits]


# ---------------------------------------------------------------------------
# Self-test: python -m src.retrieval.knn
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("KNN Retriever — self-test")
    print("=" * 60)

    import os
    from dotenv import load_dotenv
    from src.indexing.opensearch_client import get_client

    load_dotenv()
    client = get_client()
    index_name = os.getenv("OPENSEARCH_INDEX", "")

    print("\nLoading encoder ...")
    enc = Encoder()

    retriever = KNNRetriever(client, index_name, encoder=enc)

    test_query = "obstructive sleep apnea treatment"
    print(f"\nQuery: '{test_query}'")
    results = retriever.search(test_query, size=100)

    assert isinstance(results, list), "Result must be a list"
    assert len(results) == 100, f"Expected 100 results, got {len(results)}"
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True), "Scores not sorted descending"
    pmids = [p for p, _ in results]
    assert len(set(pmids)) == len(pmids), "Duplicate PMIDs in results"

    print(f"  Results count : {len(results)}  OK")
    print(f"  Score order   : descending  OK")
    print(f"  No duplicates : OK")
    print(f"\n  Top 3 results:")
    for pmid, score in results[:3]:
        print(f"    PMID={pmid}  score={score:.4f}")

    print("\n  All KNN assertions passed.")
