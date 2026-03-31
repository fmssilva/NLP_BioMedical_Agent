"""
src/retrieval/bm25.py

BM25 retriever — match query on the `contents` field (BM25 similarity).

Pattern from Lab01 search cell (lines 297-319): match query, size explicit, _source doc_id only.
"""

import os
import sys
from pathlib import Path

from opensearchpy import OpenSearch

from src.retrieval.base import BaseRetriever


class BM25Retriever(BaseRetriever):
    """BM25 retrieval using OpenSearch's built-in BM25 similarity on the `contents` field."""

    def __init__(self, client: OpenSearch, index_name: str):
        self.client = client
        self.index_name = index_name

    # Run BM25 match query on the contents field; returns top-size (pmid, score) pairs.
    def search(self, query: str, size: int = 100) -> list[tuple[str, float]]:
        query_body = {
            "size": size,
            "_source": ["doc_id"],
            "query": {
                "match": {
                    "contents": {
                        "query": query
                    }
                }
            },
        }
        response = self.client.search(body=query_body, index=self.index_name)
        hits = response["hits"]["hits"]
        return [(h["_source"]["doc_id"], h["_score"]) for h in hits]


# ---------------------------------------------------------------------------
# Self-test: python -m src.retrieval.bm25
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("BM25 Retriever — self-test")
    print("=" * 60)

    import os
    from dotenv import load_dotenv
    from src.indexing.opensearch_client import get_client

    load_dotenv()
    client = get_client()
    index_name = os.getenv("OPENSEARCH_INDEX", "")

    retriever = BM25Retriever(client, index_name)

    test_query = "obstructive sleep apnea treatment"
    print(f"\nQuery: '{test_query}'")
    results = retriever.search(test_query, size=100)

    # basic assertions
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

    print("\n  All BM25 assertions passed.")
