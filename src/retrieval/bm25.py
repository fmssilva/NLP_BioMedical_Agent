
from opensearchpy import OpenSearch

from src.indexing.index_builder import float_tag
from src.retrieval.base import SparseRetriever


class BM25Retriever(SparseRetriever):
    """
    BM25 retrieval on a pre-built contents_bm25_k{k1tag}_b{btag} field.
    All (k1, b) combinations follow the same naming -- no special 'contents' default.
    """

    _DEFAULT_K1 = 1.2
    _DEFAULT_B  = 0.75

    def __init__(
        self,
        client: OpenSearch,
        index_name: str,
        k1: float = _DEFAULT_K1,
        b:  float = _DEFAULT_B,
    ):
        field = f"contents_bm25_k{float_tag(k1)}_b{float_tag(b)}"
        super().__init__(client, index_name, field=field)


#################################################################
##                  LOCAL TEST                                 ##
#################################################################
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
    print(f"  field : {retriever.field}")

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

    print("\n  All BM25 assertions passed.")


