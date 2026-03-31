"""
src/retrieval/lm_dirichlet.py

LM Dirichlet retriever — match query on the `contents_lmdir` field (mu=2000).

Dirichlet smoothing with mu=2000 is the standard default for passage retrieval.
Same match-query pattern as BM25, different field.
"""

import os
import sys
from pathlib import Path

from opensearchpy import OpenSearch

from src.retrieval.base import BaseRetriever


class LMDirichletRetriever(BaseRetriever):
    """LM Dirichlet retrieval using the pre-configured `contents_lmdir` field (mu=2000)."""

    def __init__(self, client: OpenSearch, index_name: str):
        self.client = client
        self.index_name = index_name

    # Run LM-Dirichlet match query on the lmdir field; same pattern as BM25 but different field.
    def search(self, query: str, size: int = 100) -> list[tuple[str, float]]:
        query_body = {
            "size": size,
            "_source": ["doc_id"],
            "query": {
                "match": {
                    "contents_lmdir": {
                        "query": query
                    }
                }
            },
        }
        response = self.client.search(body=query_body, index=self.index_name)
        hits = response["hits"]["hits"]
        return [(h["_source"]["doc_id"], h["_score"]) for h in hits]


# ---------------------------------------------------------------------------
# Self-test: python -m src.retrieval.lm_dirichlet
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("LM Dirichlet Retriever — self-test")
    print("=" * 60)

    import os
    from dotenv import load_dotenv
    from src.indexing.opensearch_client import get_client

    load_dotenv()
    client = get_client()
    index_name = os.getenv("OPENSEARCH_INDEX", "")

    retriever = LMDirichletRetriever(client, index_name)

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

    print("\n  All LM-Dirichlet assertions passed.")
