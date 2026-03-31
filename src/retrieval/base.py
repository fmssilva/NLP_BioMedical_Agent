"""
src/retrieval/base.py

Abstract base class for all retrieval strategies.

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
