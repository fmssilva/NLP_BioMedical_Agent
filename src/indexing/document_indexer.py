
import os
import sys
from pathlib import Path

import numpy as np
from opensearchpy import OpenSearch
from opensearchpy.helpers import bulk
from tqdm import tqdm

from src.indexing.opensearch_client import get_client, check_health, check_index
from src.indexing.index_builder import get_live_fields, get_live_field_types


######################################################################
## Document indexing - build the full index into OpenSearch.
######################################################################



def _build_source(
    doc: dict,
    text_fields: list[str],
    knn_fields: dict[str, np.ndarray],
    doc_idx: int,
) -> dict:
    """
    Build the _source dict for a single document.
    Args:
        doc:         {"id": str, "contents": str}
        text_fields: list of text field names to populate with doc["contents"]
        knn_fields:  {field_name: full_array} — row [doc_idx] is used
        doc_idx:     position of this doc in the corpus list (= row index in arrays)
    """
    source: dict = {"doc_id": doc["id"]}

    text = doc["contents"]
    for fname in text_fields:
        source[fname] = text

    for fname, arr in knn_fields.items():
        source[fname] = arr[doc_idx].tolist()

    return source




def _get_indexed_fields(client: OpenSearch, index_name: str, doc_id: str) -> set[str]:
    """
    Return the set of field names stored in the _source of one indexed doc.
    Returns empty set if the doc doesn't exist yet.
    """
    try:
        resp = client.get(index=index_name, id=doc_id)
        return set(resp["_source"].keys())
    except Exception:
        return set()




def index_documents(
    client:         OpenSearch,
    index_name:     str,
    corpus:         list[dict],
    embeddings:     "list[tuple[str, str, np.ndarray]] | dict[str, np.ndarray]",
    batch_size:     int = 100,
) -> None:
    """
    Bulk-index all corpus documents.

    Skipped when BOTH conditions hold:
      (a) doc count == len(corpus)
      (b) all fields we're about to write are already in the first stored doc's _source
    """
    # build field_name -> array dict
    if isinstance(embeddings, dict):
        # caller already built the field name -> array mapping directly
        embeddings_map: dict[str, np.ndarray] = embeddings
    else:
        # list of (alias, model_name, vecs) tuples from create_embeddings()
        # all aliases produce embedding_{alias} (e.g. embedding_msmarco, embedding_medcpt)
        embeddings_map = {
            f"embedding_{alias}": vecs
            for alias, _model_name, vecs in embeddings
        }

    # validate shapes
    for fname, arr in embeddings_map.items():
        if len(arr) != len(corpus):
            raise ValueError(
                f"embeddings['{fname}'] has {len(arr)} rows "
                f"but corpus has {len(corpus)} docs."
            )

    # determine field types from live mapping
    # knn_vector fields not in embeddings_map are left un-populated (different encoder pass)
    live_field_types = get_live_field_types(client, index_name)
    all_knn_in_mapping = {
        fname for fname, ftype in live_field_types.items() if ftype == "knn_vector"
    }
    knn_field_names = set(embeddings_map.keys())   # what we're writing today
    text_fields = sorted(
        f for f, ftype in live_field_types.items()
        if ftype != "knn_vector" and f != "doc_id"
    )

    # expected_fields = what this call will write into each doc's _source
    expected_fields = {"doc_id"} | set(text_fields) | knn_field_names

    # check cnditions to skip: 
    #   (a) doc count in index == len(corpus)
    #       - count < corpus  -> partial index or first run -> re-index
    #       - count > corpus  -> common in test mode (CORPUS_SIZE=10 on a full 4194-doc index)
    #                      -> re-index (upsert): only the 10 given docs are updated;
    #   (b) the first stored doc already has all expected fields
    count_resp    = client.count(index=index_name)
    current_count = count_resp.get("count", 0)

    upsert_partial = current_count > len(corpus)

    if current_count == len(corpus):
        # count matches — sample the first doc's stored fields to check if any field is missing
        first_id       = corpus[0]["id"]
        stored_fields  = _get_indexed_fields(client, index_name, first_id)
        missing_in_doc = expected_fields - stored_fields

        if not missing_in_doc:
            print(
                f"[document_indexer] Already fully indexed -- "
                f"{current_count} docs, all {len(expected_fields)} fields present. Skipping."
            )
            return

        # some fields exist in the mapping but not in stored docs yet (e.g. new KNN field added)
        print(
            f"[document_indexer] New fields not yet in stored docs: {sorted(missing_in_doc)}"
            f" -- full re-index required."
        )
    elif upsert_partial:
        # index has MORE docs than corpus — common when CORPUS_SIZE is set small in the notebook
        # upsert only the given slice; the rest of the index keeps its existing data
        print(
            f"[document_indexer] Index has {current_count} docs but corpus has {len(corpus)}. "
            f"Running in partial mode — will upsert {len(corpus)} docs in-place."
        )
    elif current_count > 0:
        print(
            f"[document_indexer] Partial index: {current_count}/{len(corpus)} docs present. "
            f"Proceeding with full re-index (duplicates update in-place)."
        )
    else:
        print(f"[document_indexer] Indexing {len(corpus)} documents into '{index_name}' ...")

    # text_fields = all BM25 / LM-JM / LM-Dir fields — every sparse model has its own field
    print(f"[document_indexer] Sparse text fields ({len(text_fields)}): {text_fields}")
    print(f"[document_indexer] KNN vector fields  ({len(knn_field_names)}): {sorted(knn_field_names)}")
    if all_knn_in_mapping - knn_field_names:
        # other encoder fields exist in the mapping but aren't being written this call — that's fine
        print(
            f"[document_indexer] KNN fields in mapping but skipped this pass "
            f"(different encoder run): {sorted(all_knn_in_mapping - knn_field_names)}"
        )

    # Bulk-index in batches
    total_indexed = 0
    with tqdm(total=len(corpus), desc="Indexing documents", unit="doc") as pbar:
        for start in range(0, len(corpus), batch_size):
            batch_corpus = corpus[start: start + batch_size]

            actions = []
            for i_local, doc in enumerate(batch_corpus):
                doc_idx = start + i_local
                source  = _build_source(doc, text_fields, embeddings_map, doc_idx)
                actions.append({
                    "_index": index_name,
                    "_id":    doc["id"],   # PMID as OpenSearch doc ID
                    "_source": source,
                })

            success, errors = bulk(client, actions, raise_on_error=False)
            total_indexed += success
            pbar.update(len(batch_corpus))

            if errors:
                print(f"\n[document_indexer] {len(errors)} errors in batch starting at {start}:")
                for err in errors[:3]:
                    print(f"  {err}")

    # Refresh so documents are immediately searchable
    client.indices.refresh(index=index_name)

    final_count = client.count(index=index_name).get("count", 0)
    suffix = f" (partial upsert — corpus slice={len(corpus)})" if upsert_partial else f"/{len(corpus)}"
    print(f"[document_indexer] Done.  Indexed: {total_indexed}  |  Total in index: {final_count}{suffix}")
    if not upsert_partial and final_count != len(corpus):
        print(
            f"[document_indexer] WARNING — final count {final_count} != {len(corpus)}. "
            "Some documents may have failed to index."
        )



#################################################################
##                  LOCAL TEST                                 ##
#################################################################
if __name__ == "__main__":
    print("=" * 60)
    print("document_indexer.py — self-test")
    print("=" * 60)

    root = Path(__file__).resolve().parents[2]

    print("\n[1/5] Connecting to OpenSearch …")
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

    print("\n[2/5] Loading corpus …")
    from src.data.loader import load_corpus
    corpus = load_corpus(root / "data" / "filtered_pubmed_abstracts.txt")
    print(f"      Corpus: {len(corpus)} docs")

    print("\n[3/5] Loading embeddings …")
    from src.embeddings.corpus_encoder import load_embeddings
    emb_path = root / "embeddings" / "pubmed_knn_vectors.npy"
    if not emb_path.exists():
        print(f"      ERROR: not found at {emb_path}")
        print("      Run: python -m src.embeddings.corpus_encoder")
        sys.exit(1)
    emb = load_embeddings(emb_path)
    assert emb.shape == (len(corpus), 768), f"Expected ({len(corpus)}, 768), got {emb.shape}"
    print(f"      Shape: {emb.shape}  OK")

    print("\n[4/5] Doc count before …")
    count_before = os_client.count(index=index_name).get("count", 0)
    print(f"      Count: {count_before}")

    print("\n[5/5] Running index_documents() …")
    index_documents(
        os_client,
        index_name,
        corpus,
        {"embedding_msmarco": emb},
    )

    count_after = os_client.count(index=index_name).get("count", 0)
    print(f"\n      Count after: {count_after}")
    assert count_after == len(corpus), f"Expected {len(corpus)}, got {count_after}"
    print("      OK")

    check_index(os_client, index_name)

    print("\n" + "=" * 60)
    print("document_indexer.py  —  all tests passed [ok]")
    print("=" * 60)
