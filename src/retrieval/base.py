
# -*- coding: utf-8 -*-
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

######################################################################
## Base retriever classes — shared interface and sparse search impl.
######################################################################

def _extract_hits(hits: list[dict]) -> list[tuple[str, float]]:
    """
    Extracts (doc_id, score) tuples from OpenSearch hits.
    Skips hits that are missing doc_id (logs a warning).
    """
    results = []
    for h in hits:
        doc_id = h.get("_source", {}).get("doc_id")
        if doc_id is None:
            logger.warning("[retrieval] Hit missing doc_id — skipping. _id=%s", h.get("_id"))
            continue
        results.append((doc_id, h["_score"]))
    return results


class BaseRetriever(ABC):
    """
    Interface for all retrieval strategies (Sparse and Dense Knn).
    This uniform interface makes evaluation and RRF fusion easy.

    Hierarchy:
        BaseRetriever:          ← abstract interface (+ field_exists utility for everyone)
            |SparseRetriever      ← concrete shared impl for BM25 / LM-JM / LM-Dir
                |BM25Retriever
                |LMJMRetriever
                |LMDirichletRetriever
            |KNNRetriever         ← dense vector search, very different query body
            |RRFRetriever         ← hybrid: fuses a sparse + dense ranked list
    """

    @abstractmethod
    def search(self, query: str, size: int = 100) -> list[tuple[str, float]]:
        """Return (pmid, score) pairs sorted by score descending."""
        pass


    @classmethod
    def field_exists(cls, client, index_name: str, field: str) -> bool:
        """
        Return True if ``field`` is present in the live index mapping.
        """
        if not client.indices.exists(index=index_name):
            return False
        resp = client.indices.get_mapping(index=index_name)
        actual_key = list(resp.keys())[0]
        props = resp[actual_key]["mappings"].get("properties", {})
        return field in props

    @classmethod
    def assert_field_exists(cls, client, index_name: str, field: str) -> None:
        """
        Raise ValueError with a clear message if ``field`` is not in the index.
        """
        if not cls.field_exists(client, index_name, field):
            raise ValueError(
                f"Field '{field}' not found in index '{index_name}'. "
                "Was the index built with the right model params? "
                "Check index_builder.build_index_mapping()."
            )



class SparseRetriever(BaseRetriever):
    """
    Shared implementation for all sparse retrieval models: BM25, LM-JM, LM-Dir.
    Sends a standard ``match`` (or ``match_phrase``) query to the named text field.
    Each model family uses a dedicated named field baked at indexing time.
    """

    # Example of query types that work in opensearch for text fields (sparce models)
    _VALID_MATCH_TYPES = ("match", "match_phrase")


    def __init__(self, client, index_name: str, field: str,
                 match_type: str = "match"):
        if match_type not in self._VALID_MATCH_TYPES:
            raise ValueError(
                f"match_type must be one of {self._VALID_MATCH_TYPES}, got '{match_type}'"
            )
        self.client = client
        self.index_name = index_name
        self.field = field
        self.match_type = match_type


    def _build_body(self, query: str, size: int) -> dict:
        """Build the OpenSearch request body for this field and match type."""
        return {
            "size": size,
            "_source": ["doc_id"],
            "query": {self.match_type: {self.field: {"query": query}}},
        }

    def search(self, query: str, size: int = 100) -> list[tuple[str, float]]:
        body = self._build_body(query, size)
        resp = self.client.search(body=body, index=self.index_name)
        return _extract_hits(resp["hits"]["hits"])
