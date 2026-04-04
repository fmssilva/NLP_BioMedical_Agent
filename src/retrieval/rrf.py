
from src.retrieval.base import BaseRetriever


def rrf_merge(
    run_a: list[tuple[str, float]],
    run_b: list[tuple[str, float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Merge two ranked lists with RRF (Cormack 2009). score = 1/(k+rank_a) + 1/(k+rank_b)."""
    scores: dict[str, float] = {}
    for rank, (doc_id, _) in enumerate(run_a, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    for rank, (doc_id, _) in enumerate(run_b, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class RRFRetriever(BaseRetriever):
    """Fuses any two BaseRetriever instances with RRF."""

    def __init__(
        self,
        retriever_a: BaseRetriever,
        retriever_b: BaseRetriever,
        rrf_k: int = 60,
    ):
        if not isinstance(retriever_a, BaseRetriever):
            raise TypeError(f"retriever_a must be a BaseRetriever, got {type(retriever_a)}")
        if not isinstance(retriever_b, BaseRetriever):
            raise TypeError(f"retriever_b must be a BaseRetriever, got {type(retriever_b)}")
        self.retriever_a = retriever_a
        self.retriever_b = retriever_b
        self.rrf_k = rrf_k

    def search(self, query: str, size: int = 100) -> list[tuple[str, float]]:
        run_a = self.retriever_a.search(query, size=size)
        run_b = self.retriever_b.search(query, size=size)
        return rrf_merge(run_a, run_b, k=self.rrf_k)[:size]


#################################################################
##                  LOCAL TEST                                 ##
#################################################################
if __name__ == "__main__":
    print("=" * 60)
    print("RRF Retriever — self-test")
    print("=" * 60)

    import os
    from dotenv import load_dotenv
    from src.indexing.opensearch_client import get_client
    from src.embeddings.encoder import Encoder
    from src.retrieval.bm25 import BM25Retriever
    from src.retrieval.knn import KNNRetriever
    from src.retrieval.lm_jelinek_mercer import LMJMRetriever
    from src.retrieval.lm_dirichlet import LMDirichletRetriever

    load_dotenv()
    client = get_client()
    index_name = os.getenv("OPENSEARCH_INDEX", "")
    query = "obstructive sleep apnea treatment"

    print("\nLoading encoder ...")
    enc = Encoder()

    pairs = [
        ("BM25 + KNN",   BM25Retriever(client, index_name),
                         KNNRetriever(client, index_name, encoder=enc)),
        ("BM25 + LMJM",  BM25Retriever(client, index_name),
                         LMJMRetriever(client, index_name, lambd=0.7)),
        ("BM25 + LMDir", BM25Retriever(client, index_name),
                         LMDirichletRetriever(client, index_name, mu=75)),
        ("LMJM + LMDir", LMJMRetriever(client, index_name, lambd=0.7),
                         LMDirichletRetriever(client, index_name, mu=75)),
    ]

    for label, a, b in pairs:
        print(f"\n-- {label} --")
        rrf = RRFRetriever(a, b)
        results = rrf.search(query, size=100)
        scores = [s for _, s in results]
        pmids  = [p for p, _ in results]
        assert len(results) == 100, f"Expected 100, got {len(results)}"
        assert scores == sorted(scores, reverse=True), "Not sorted"
        assert len(set(pmids)) == len(pmids), "Duplicates"
        assert all(0 < s < 1 for _, s in results), "Scores out of range"
        print(f"  count=100  sorted  no-dupes  scores in (0,1)  OK")
        print(f"  top-1: PMID={pmids[0]}  score={scores[0]:.6f}")

    print("\n  All RRF assertions passed.")
