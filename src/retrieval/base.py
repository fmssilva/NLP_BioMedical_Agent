"""
src/retrieval/base.py

Abstract base class for all retrieval strategies + generic FieldRetriever.

All retrievers return a ranked list of (pmid, score) tuples, highest score first,
with size=100 by default. This interface makes evaluation and RRF fusion uniform.
"""

from abc import ABC, abstractmethod


class BaseRetriever(ABC):
    # Return ranked list of (pmid, score) pairs for the given query.
    @abstractmethod
    def search(self, query: str, size: int = 100) -> list[tuple[str, float]]:
        """
        Run a retrieval query.

        Args:
            query: query string (already formatted — field concatenation happens upstream).
            size:  max number of results to return. Always pass 100 explicitly; default is here
                   for convenience but OpenSearch returns 10 without it.

        Returns:
            List of (pmid, score) tuples, sorted by score descending.
            Never returns duplicates. Length <= size.
        """
        pass


class FieldRetriever(BaseRetriever):
    """
    Generic retriever: runs a match query on any named text field.

    Works for BM25, LM-JM, LM-Dir, and any tuning-sweep fields — the scoring
    function is baked into the field at index time, so the query is always
    the same match query regardless of the similarity algorithm.

    Used by: final_eval.py (tuned BM25, tuned LM-Dir), lmdir_mu_sweep.py,
             bm25_param_sweep.py.
    """

    def __init__(self, client, index_name: str, field: str):
        self.client = client
        self.index_name = index_name
        self.field = field

    def search(self, query: str, size: int = 100) -> list[tuple[str, float]]:
        body = {
            "size": size,
            "_source": ["doc_id"],
            "query": {"match": {self.field: {"query": query}}},
        }
        resp = self.client.search(body=body, index=self.index_name)
        return [(h["_source"]["doc_id"], h["_score"]) for h in resp["hits"]["hits"]]
