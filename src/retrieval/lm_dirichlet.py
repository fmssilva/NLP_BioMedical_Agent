
from opensearchpy import OpenSearch

from src.retrieval.base import SparseRetriever


class LMDirichletRetriever(SparseRetriever):
    """
    LM Dirichlet retrieval on a pre-built contents_lmdir_{mu} field.
    """

    def __init__(self, client: OpenSearch, index_name: str, mu: int = 2000):
        if mu <= 0:
            raise ValueError(f"mu must be a positive integer, got {mu}")
        field = f"contents_lmdir_{mu}"
        super().__init__(client, index_name, field=field)


#################################################################
##                  LOCAL TEST                                 ##
#################################################################
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

    retriever = LMDirichletRetriever(client, index_name, mu=75)
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

    print("\n  All LM-Dirichlet assertions passed.")


