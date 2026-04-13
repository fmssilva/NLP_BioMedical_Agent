
import numpy as np
from opensearchpy import OpenSearch

from src.embeddings.encoder import Encoder
from src.retrieval.base import BaseRetriever, _extract_hits

_DEFAULT_ALIAS = "msmarco"


def _embedding_field(alias: str) -> str:
    """Derive the OpenSearch field name: all aliases get embedding_{alias}."""
    return f"embedding_{alias}"


class KNNRetriever(BaseRetriever):
    """
    Dense KNN retrieval on an embedding_{alias} field.
    """

    def __init__(
        self,
        client: OpenSearch,
        index_name: str,
        encoder: "Encoder | None" = None,
        encoder_alias: str = _DEFAULT_ALIAS,
    ):
        if encoder is not None and not hasattr(encoder, "encode_single"):
            raise TypeError(
                f"encoder must be an Encoder instance (or duck-typed equivalent) or None, "
                f"got {type(encoder)}"
            )
        self.client = client
        self.index_name = index_name
        self.encoder = encoder if encoder is not None else Encoder()
        self.encoder_alias = encoder_alias
        self.embed_field = _embedding_field(encoder_alias)

    def search(self, query: str, size: int = 100) -> list:
        """Encode query -> vector, then run knn search on self.embed_field. k == size per Lab01."""
        query_vector = self.encoder.encode_single(query)  # shape (dim,)

        body = {
            "size": size,
            "_source": ["doc_id"],
            "query": {
                "knn": {
                    self.embed_field: {
                        "vector": query_vector.tolist(),
                        "k": size,
                    }
                }
            },
        }
        resp = self.client.search(body=body, index=self.index_name)
        return _extract_hits(resp["hits"]["hits"])


# ---------------------------------------------------------------------------
# MedCPT asymmetric KNN retriever
# ---------------------------------------------------------------------------

class MedCPTKNNRetriever:
    """
    Dense KNN retrieval using MedCPT embeddings stored in the index.

    MedCPT uses an asymmetric architecture — a separate query encoder
    (ncbi/MedCPT-Query-Encoder) from the document encoder.  The query encoder
    is loaded lazily on first use.
    """

    def __init__(self, client, index_name: str):
        self.client = client
        self.index_name = index_name
        self._tokenizer = None
        self._model = None
        self._device = "cpu"

    def _ensure_encoder(self) -> None:
        """Load MedCPT query encoder on first use."""
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        model_name = "ncbi/MedCPT-Query-Encoder"
        print(f"[MedCPTKNNRetriever] Loading '{model_name}' ...")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).eval()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)

    def encode_query(self, text: str) -> "np.ndarray":
        """Encode a single query; returns (768,) L2-normalised vector.

        MedCPT-Query-Encoder uses CLS pooling (last_hidden_state[:, 0, :]) with
        max_length=64, matching the official NCBI usage example.
        """
        import torch
        import torch.nn.functional as F

        self._ensure_encoder()
        enc = self._tokenizer(
            [text], padding=True, truncation=True,
            max_length=64, return_tensors="pt",
        )
        enc = {k: v.to(self._device) for k, v in enc.items()}
        with torch.no_grad():
            out = self._model(**enc, return_dict=True)
        # CLS token — official MedCPT pooling strategy
        cls_vec = out.last_hidden_state[:, 0, :]
        normed = F.normalize(cls_vec, p=2, dim=1)
        return normed[0].cpu().numpy()

    def search(self, query: str, size: int = 100) -> list[tuple[str, float]]:
        qvec = self.encode_query(query)
        body = {
            "size": size,
            "_source": ["doc_id"],
            "query": {
                "knn": {
                    "embedding_medcpt": {
                        "vector": qvec.tolist(),
                        "k": size,
                    }
                }
            },
        }
        resp = self.client.search(body=body, index=self.index_name)
        return [(h["_source"]["doc_id"], h["_score"]) for h in resp["hits"]["hits"]]


#################################################################
##                  LOCAL TEST                                 ##
#################################################################
if __name__ == "__main__":
    print("=" * 60)
    print("KNN Retriever -- self-test")
    print("=" * 60)

    import os
    from dotenv import load_dotenv
    from src.indexing.opensearch_client import get_client

    load_dotenv()
    client = get_client()
    index_name = os.getenv("OPENSEARCH_INDEX", "")

    print("\nLoading encoder ...")
    enc = Encoder()

    retriever = KNNRetriever(client, index_name, encoder=enc)
    print(f"  embed_field : {retriever.embed_field}")

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

    print("\n  All KNN assertions passed.")
