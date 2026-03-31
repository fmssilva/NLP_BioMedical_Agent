"""
src/retrieval/lm_jelinek_mercer.py

LM Jelinek-Mercer retriever — match query on a pre-configured field.

Two variants are baked into the index at creation time:
  - contents_lmjm_01  (lambda=0.1 — favours short queries, document-centric smoothing)
  - contents_lmjm_07  (lambda=0.7 — favours longer queries, corpus-centric smoothing)

Select with lambda_variant="01" or "07". Compare both on train set, lock winner.
"""

import os
import sys
from pathlib import Path

from opensearchpy import OpenSearch

from src.retrieval.base import BaseRetriever

# field name template — variant is "01" or "07"
_FIELD_MAP = {
    "01": "contents_lmjm_01",
    "07": "contents_lmjm_07",
}


class LMJMRetriever(BaseRetriever):
    """LM Jelinek-Mercer retrieval on one of the two pre-configured LM-JM fields."""

    def __init__(self, client: OpenSearch, index_name: str, lambda_variant: str = "01"):
        """
        Args:
            lambda_variant: "01" for lambda=0.1, "07" for lambda=0.7.
        """
        if lambda_variant not in _FIELD_MAP:
            raise ValueError(f"lambda_variant must be '01' or '07', got '{lambda_variant}'")
        self.client = client
        self.index_name = index_name
        self.lambda_variant = lambda_variant
        self.field = _FIELD_MAP[lambda_variant]

    # Run LM-JM match query on the appropriate field; same pattern as BM25 but different field.
    def search(self, query: str, size: int = 100) -> list[tuple[str, float]]:
        query_body = {
            "size": size,
            "_source": ["doc_id"],
            "query": {
                "match": {
                    self.field: {
                        "query": query
                    }
                }
            },
        }
        response = self.client.search(body=query_body, index=self.index_name)
        hits = response["hits"]["hits"]
        return [(h["_source"]["doc_id"], h["_score"]) for h in hits]


# ---------------------------------------------------------------------------
# Self-test: python -m src.retrieval.lm_jelinek_mercer
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("LM Jelinek-Mercer Retriever — self-test")
    print("=" * 60)

    import os
    from dotenv import load_dotenv
    from src.indexing.opensearch_client import get_client

    load_dotenv()
    client = get_client()
    index_name = os.getenv("OPENSEARCH_INDEX", "")

    test_query = "obstructive sleep apnea treatment"
    print(f"\nQuery: '{test_query}'")

    for variant in ["01", "07"]:
        retriever = LMJMRetriever(client, index_name, lambda_variant=variant)
        results = retriever.search(test_query, size=100)

        assert isinstance(results, list), "Result must be a list"
        assert len(results) == 100, f"Expected 100 results, got {len(results)}"
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True), "Scores not sorted descending"
        pmids = [p for p, _ in results]
        assert len(set(pmids)) == len(pmids), "Duplicate PMIDs in results"

        print(f"\n  lambda={variant}  field={retriever.field}")
        print(f"  Results count : {len(results)}  OK")
        print(f"  Score order   : descending  OK")
        print(f"  No duplicates : OK")
        print(f"  Top 3:")
        for pmid, score in results[:3]:
            print(f"    PMID={pmid}  score={score:.4f}")

    print("\n  All LM-JM assertions passed.")
