"""
src/indexing/document_indexer.py

Bulk-index all 4194 PubMed abstracts into OpenSearch.

Public API:
    index_documents(client, index_name, corpus, embeddings) -> None

Pattern from Lab03 bulk indexing + Lab01 document structure.
Idempotent: if doc count already equals len(corpus), skips indexing entirely.
Each document gets 5 text fields (BM25 + 2x LM-JM + LM-Dir + same contents for all)
plus one knn_vector embedding field.
"""

import os
import sys
from pathlib import Path

import numpy as np
from opensearchpy import OpenSearch
from opensearchpy.helpers import bulk
from tqdm import tqdm

from src.indexing.opensearch_client import get_client, check_health, check_index


# Index all documents, skipping if already indexed. Refreshes index when done.
def index_documents(
    client: OpenSearch,
    index_name: str,
    corpus: list[dict],
    embeddings: np.ndarray,
    batch_size: int = 100,
) -> None:
    """
    Bulk-index all corpus documents with their embeddings.

    Idempotency: checks doc count first. If count == len(corpus), skips indexing.
    Each action sets all 5 text fields to the same 'contents' string — the index
    mapping applies different similarities to each field at query time.

    Args:
        client:     Connected OpenSearch client.
        index_name: Name of the target index (must already exist with correct mapping).
        corpus:     List of {"id": str, "contents": str} dicts (4194 docs).
        embeddings: np.ndarray shape (len(corpus), 768) — L2-normalised vectors.
        batch_size: Number of documents per bulk request.
    """
    if len(corpus) != len(embeddings):
        raise ValueError(
            f"corpus ({len(corpus)}) and embeddings ({len(embeddings)}) must have the same length"
        )

    # idempotency check — if already indexed, skip
    count_resp = client.count(index=index_name)
    current_count = count_resp.get("count", 0)
    if current_count == len(corpus):
        print(f"[document_indexer] Already indexed {current_count} docs — skipping.")
        return

    if current_count > 0:
        print(
            f"[document_indexer] Partial index: {current_count}/{len(corpus)} docs present. "
            f"Proceeding with full re-index (duplicates will update in-place)."
        )
    else:
        print(f"[document_indexer] Indexing {len(corpus)} documents into '{index_name}' ...")

    # build bulk actions in batches and send
    total_indexed = 0
    with tqdm(total=len(corpus), desc="Indexing documents", unit="doc") as pbar:
        for start in range(0, len(corpus), batch_size):
            batch_corpus = corpus[start: start + batch_size]
            batch_embs   = embeddings[start: start + batch_size]

            actions = []
            for doc, emb in zip(batch_corpus, batch_embs):
                text = doc["contents"]
                actions.append({
                    "_index": index_name,
                    "_id":    doc["id"],   # use PMID as the OpenSearch document ID
                    "_source": {
                        "doc_id":           doc["id"],
                        "contents":         text,
                        "contents_lmjm_01": text,   # same text, different similarity field
                        "contents_lmjm_07": text,
                        "contents_lmdir":   text,
                        "embedding":        emb.tolist(),
                    },
                })

            success, errors = bulk(client, actions, raise_on_error=False)
            total_indexed += success
            pbar.update(len(batch_corpus))

            if errors:
                # print each error but continue — partial indexing is recoverable
                print(f"\n[document_indexer] {len(errors)} errors in batch starting at {start}:")
                for err in errors[:3]:   # show at most 3 per batch
                    print(f"  {err}")

    # refresh so documents are immediately searchable
    client.indices.refresh(index=index_name)

    final_count = client.count(index=index_name).get("count", 0)
    print(
        f"[document_indexer] Done. "
        f"Indexed: {total_indexed}  |  Index doc count: {final_count}/{len(corpus)}"
    )
    if final_count != len(corpus):
        print(
            f"[document_indexer] WARNING — final count {final_count} != {len(corpus)}. "
            "Some documents may have failed to index."
        )


# ---------------------------------------------------------------------------
# Self-test: python -m src.indexing.document_indexer
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Step 10 — document_indexer.py self-test")
    print("=" * 60)

    root = Path(__file__).resolve().parents[2]

    # 1. Connect
    print("\n[1/5] Connecting to OpenSearch ...")
    try:
        os_client = get_client()
        print("      Client created.")
    except ValueError as e:
        print(f"      ERROR: {e}")
        sys.exit(1)

    check_health(os_client)

    from dotenv import load_dotenv
    load_dotenv()
    index_name = os.getenv("OPENSEARCH_INDEX", "")
    if not index_name:
        print("ERROR: OPENSEARCH_INDEX not set in .env")
        sys.exit(1)

    # 2. Load corpus
    print("\n[2/5] Loading corpus ...")
    from src.data.loader import load_corpus
    corpus = load_corpus(root / "data" / "filtered_pubmed_abstracts.txt")
    print(f"      Corpus: {len(corpus)} docs")

    # 3. Load embeddings
    print("\n[3/5] Loading embeddings ...")
    from src.embeddings.corpus_encoder import load_embeddings
    embeddings_path = root / "embeddings" / "pubmed_knn_vectors.npy"
    if not embeddings_path.exists():
        print(f"      ERROR: embeddings not found at {embeddings_path}")
        print("      Run: python -m src.embeddings.corpus_encoder")
        sys.exit(1)
    embeddings = load_embeddings(embeddings_path)
    assert embeddings.shape == (len(corpus), 768), (
        f"Expected ({len(corpus)}, 768), got {embeddings.shape}"
    )
    print(f"      Embeddings shape: {embeddings.shape}  OK")

    # 4. Check doc count before
    print("\n[4/5] Checking doc count before indexing ...")
    count_before = os_client.count(index=index_name).get("count", 0)
    print(f"      Doc count before: {count_before}")

    # 5. Index (idempotent)
    print("\n[5/5] Running index_documents() ...")
    index_documents(os_client, index_name, corpus, embeddings)

    count_after = os_client.count(index=index_name).get("count", 0)
    print(f"\n      Doc count after: {count_after}")
    assert count_after == len(corpus), (
        f"Expected {len(corpus)} docs, got {count_after}"
    )
    print(f"      Count matches corpus size ({len(corpus)})  OK")

    # confirm index is fully populated
    check_index(os_client, index_name)

    print("\n" + "=" * 60)
    print("document_indexer.py  —  all tests passed")
    print("=" * 60)
