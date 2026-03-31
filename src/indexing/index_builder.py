"""
src/indexing/index_builder.py

OpenSearch index creation for the BioGen IR pipeline.

Public API:
    build_index_mapping() -> dict
    create_index(client, index_name, mapping) -> None

The index holds all 4194 PubMed abstracts with:
  - BM25 text field            (contents)
  - LM Jelinek-Mercer λ=0.1    (contents_lmjm_01)
  - LM Jelinek-Mercer λ=0.7    (contents_lmjm_07)
  - LM Dirichlet μ=2000        (contents_lmdir)
  - 768-dim KNN vector field    (embedding, HNSW faiss innerproduct)

Index creation is guarded: if the index already exists the function prints
a notice and returns — it NEVER deletes an existing index.
"""

import os
import sys

from opensearchpy import OpenSearch

from src.indexing.opensearch_client import get_client, check_health


# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------

def build_index_mapping() -> dict:
    """
    Return the full OpenSearch index body (settings + mappings).

    Key decisions:
    - `standard` analyzer (NOT `english`): Porter stemming mangles biomedical
      terms — standard tokenises + lowercases only, matching Lab01.
    - 4 shards, 0 replicas, refresh_interval=-1 (fast bulk indexing).
    - ef_search=100 in index settings (avoids per-query override issues with
      the OpenSearch KNN plugin at the server version we target).
    - HNSW faiss innerproduct: requires L2-normalised embeddings so that
      inner product == cosine similarity (msmarco-distilbert-base-v2 output
      is L2-normalised by our encoder).
    - dynamic: strict — no accidental extra fields accepted.
    """
    return {
        "settings": {
            "index": {
                "number_of_shards": 4,
                "number_of_replicas": 0,
                "refresh_interval": "-1",
                "knn": "true",
                "knn.algo_param.ef_search": 100,
            },
            "similarity": {
                "lmjm_01_similarity": {
                    "type": "LMJelinekMercer",
                    "lambda": 0.1,
                },
                "lmjm_07_similarity": {
                    "type": "LMJelinekMercer",
                    "lambda": 0.7,
                },
                "lmdir_similarity": {
                    "type": "LMDirichlet",
                    "mu": 2000,
                },
            },
        },
        "mappings": {
            "dynamic": "strict",
            "properties": {
                "doc_id": {
                    "type": "keyword",
                },
                "contents": {
                    "type": "text",
                    "analyzer": "standard",
                    "similarity": "BM25",
                },
                "contents_lmjm_01": {
                    "type": "text",
                    "analyzer": "standard",
                    "similarity": "lmjm_01_similarity",
                },
                "contents_lmjm_07": {
                    "type": "text",
                    "analyzer": "standard",
                    "similarity": "lmjm_07_similarity",
                },
                "contents_lmdir": {
                    "type": "text",
                    "analyzer": "standard",
                    "similarity": "lmdir_similarity",
                },
                "embedding": {
                    "type": "knn_vector",
                    "dimension": 768,
                    "method": {
                        "name": "hnsw",
                        "space_type": "innerproduct",
                        "engine": "faiss",
                        "parameters": {
                            "ef_construction": 256,
                            "m": 48,
                        },
                    },
                },
            },
        },
    }


# ---------------------------------------------------------------------------
# Index creation
# ---------------------------------------------------------------------------

def create_index(client: OpenSearch, index_name: str, mapping: dict) -> None:
    """
    Create the OpenSearch index with the given mapping.

    If the index already exists the function logs a message and returns
    without making any changes — it NEVER deletes an existing index.

    Args:
        client:     A connected OpenSearch client.
        index_name: Name of the index to create.
        mapping:    Full index body dict (from build_index_mapping()).
    """
    if client.indices.exists(index=index_name):
        print(f"[index_builder] Index '{index_name}' already exists — skipping creation.")
        return

    print(f"[index_builder] Creating index '{index_name}' …")
    response = client.indices.create(index=index_name, body=mapping)
    if response.get("acknowledged"):
        print(f"[index_builder] Index '{index_name}' created successfully [ok]")
    else:
        raise RuntimeError(
            f"Index creation not acknowledged for '{index_name}'. Response: {response}"
        )


# ---------------------------------------------------------------------------
# Mapping verification helpers (used in __main__ test)
# ---------------------------------------------------------------------------

def _verify_mapping(client: OpenSearch, index_name: str) -> None:
    """
    Print a compact summary of the index settings and mappings.
    Used in the __main__ test to confirm the created index matches the spec.
    """
    raw_settings = client.indices.get_settings(index=index_name)
    # The response key is the actual index name (may differ from alias)
    actual_key = list(raw_settings.keys())[0]
    settings = raw_settings[actual_key]["settings"]["index"]
    mappings = client.indices.get_mapping(index=index_name)[actual_key]["mappings"]

    # --- Settings ---
    shards   = settings.get("number_of_shards", "?")
    replicas = settings.get("number_of_replicas", "?")
    refresh  = settings.get("refresh_interval", "?")
    knn_flag = settings.get("knn", "?")
    # ef_search may be nested under "knn.algo_param.ef_search" as a flat key
    ef_search = settings.get("knn.algo_param.ef_search", "not set")

    print("\n  [settings]")
    print(f"    shards={shards}  replicas={replicas}  "
          f"refresh_interval={refresh}  knn={knn_flag}  ef_search={ef_search}")

    # --- Similarities ---
    sims = settings.get("similarity", {})
    if sims:
        print("  [similarities]")
        for name, cfg in sims.items():
            print(f"    {name}: {cfg}")
    else:
        print("  [similarities] none found in settings (expected on fresh index "
              "if similarity settings aren't echoed back by this server)")

    # --- Mappings ---
    props = mappings.get("properties", {})
    print("  [mappings]")
    for field, fdef in props.items():
        ftype = fdef.get("type", "?")
        extra = ""
        if "similarity" in fdef:
            extra = f"  similarity={fdef['similarity']}"
        if "dimension" in fdef:
            extra = f"  dimension={fdef['dimension']}"
        print(f"    {field}: type={ftype}{extra}")

    # --- Expected field list ---
    expected_fields = {
        "doc_id", "contents", "contents_lmjm_01",
        "contents_lmjm_07", "contents_lmdir", "embedding",
    }
    actual_fields = set(props.keys())
    missing = expected_fields - actual_fields
    extra   = actual_fields - expected_fields

    if not missing and not extra:
        print("\n  Fields match expected spec [ok]")
    else:
        if missing:
            print(f"\n  WARNING — missing fields: {missing}")
        if extra:
            print(f"\n  WARNING — unexpected fields: {extra}")


# ---------------------------------------------------------------------------
# Self-test: python -m src.indexing.index_builder
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Step 7 — index_builder.py self-test")
    print("=" * 60)

    # 1. Connect
    print("\n[1/4] Connecting to OpenSearch …")
    try:
        os_client = get_client()
        print("      Client created successfully.")
    except ValueError as e:
        print(f"      ERROR: {e}")
        sys.exit(1)

    # 2. Health check
    print("\n[2/4] Checking cluster health …")
    try:
        check_health(os_client)
    except RuntimeError as e:
        print(f"      ERROR: {e}")
        sys.exit(1)

    # 3. Build mapping and create index
    from dotenv import load_dotenv
    load_dotenv()
    index_name = os.getenv("OPENSEARCH_INDEX", "")
    if not index_name:
        print("\nERROR: OPENSEARCH_INDEX not set in .env — cannot create index.")
        sys.exit(1)

    print(f"\n[3/4] Building mapping and creating index '{index_name}' …")
    mapping = build_index_mapping()
    create_index(os_client, index_name, mapping)

    # 4. Verify the settings and mappings
    print(f"\n[4/4] Verifying index '{index_name}' settings and mappings …")
    try:
        _verify_mapping(os_client, index_name)
    except Exception as e:
        print(f"      WARNING: could not retrieve index details — {e}")

    print("\n" + "=" * 60)
    print("index_builder.py  —  all checks passed [ok]")
    print("=" * 60)
