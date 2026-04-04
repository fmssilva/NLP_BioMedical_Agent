
from opensearchpy import OpenSearch

from src.indexing.index_builder import float_tag
from src.retrieval.base import SparseRetriever


class LMJMRetriever(SparseRetriever):
    """
    LM Jelinek-Mercer retrieval on a pre-built contents_lmjm_{lambda_tag} field.
    """

    def __init__(self, client: OpenSearch, index_name: str, lambd: float = 0.7):
        if not (0.0 < lambd < 1.0):
            raise ValueError(f"lam must be in (0, 1), got {lambd}")
        field = f"contents_lmjm_{float_tag(lambd)}"
        super().__init__(client, index_name, field=field)


#################################################################
##                  LOCAL TEST                                 ##
#################################################################
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

    for lam in [0.1, 0.7]:
        retriever = LMJMRetriever(client, index_name, lambd=lam)
        results = retriever.search(test_query, size=100)

        assert isinstance(results, list), "Result must be a list"
        assert len(results) == 100, f"Expected 100 results, got {len(results)}"
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True), "Scores not sorted descending"
        pmids = [p for p, _ in results]
        assert len(set(pmids)) == len(pmids), "Duplicate PMIDs in results"

        print(f"\n  lam={lam}  field={retriever.field}")
        print(f"  Results count : {len(results)}  OK")
        print(f"  Score order   : descending  OK")
        print(f"  No duplicates : OK")
        print(f"  Top 3:")
        for pmid, score in results[:3]:
            print(f"    PMID={pmid}  score={score:.4f}")

    print("\n  All LM-JM assertions passed.")


