"""
src/indexing/__index_builder_test.py

Comprehensive tests for the indexing layer using a 10-doc mini-corpus.

Test flow:
  1.  Connect to OpenSearch, confirm cluster is healthy.
  2.  Delete the test index (clean slate).
  3.  build_index_mapping() dry run — no OpenSearch call.
  4.  create_or_update_index() — new index (Branch A: create).
  5.  Index 10 docs with a fake 4-dim baseline embedding.
  6.  Verify doc count = 10, all baseline text fields populated.
  7.  index_documents() idempotency — same corpus + same fields: must skip.
  8.  create_or_update_index() idempotency — same args on existing index: nothing to add.
  9.  create_or_update_index() add-fields — call with extra BM25/LM-JM/LM-Dir/KNN
      configs on the existing index (Branch B: diff + add).
  10. Full index from scratch: delete + recreate with ALL field types.
  11. Index 10 docs with 2 KNN fields.
  12. new-field detection: index_documents must re-index when a new KNN field is added.
  13. Shape mismatch guard: index_documents must raise ValueError.
  14. KNN dim mismatch warning in create_or_update_index.
  15. Clean up — delete test index.

Run with:
    cd C:\\Users\\franc\\Desktop\\NLP_Biomedical_Agent
    C:/Users/franc/anaconda3/envs/cnn/python.exe -m src.indexing.__index_builder_test
"""

import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from opensearchpy import OpenSearch

# make sure project root is on sys.path when running as __main__
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.indexing.index_builder import (
    IndexSettings,
    build_index_mapping,
    create_or_update_index,
    delete_index,
    get_live_fields,
    get_live_field_types,
)
from src.indexing.document_indexer import index_documents
from src.indexing.opensearch_client import get_client, check_health


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def verify_mapping(client: OpenSearch, index_name: str) -> None:
    """Print a compact summary of the live index settings and field mappings."""
    if not client.indices.exists(index=index_name):
        print(f"[index_builder] '{index_name}' does not exist.")
        return

    raw_settings = client.indices.get_settings(index=index_name)
    actual_key   = list(raw_settings.keys())[0]
    idx_settings = raw_settings[actual_key]["settings"]["index"]
    mappings      = client.indices.get_mapping(index=index_name)[actual_key]["mappings"]

    shards    = idx_settings.get("number_of_shards", "?")
    replicas  = idx_settings.get("number_of_replicas", "?")
    refresh   = idx_settings.get("refresh_interval", "?")
    knn_flag  = idx_settings.get("knn", "?")
    ef_search = idx_settings.get("knn.algo_param.ef_search", "not set")

    print(f"\n  [settings]  shards={shards}  replicas={replicas}  "
          f"refresh={refresh}  knn={knn_flag}  ef_search={ef_search}")

    sims = idx_settings.get("similarity", {})
    if sims:
        print("  [similarities]")
        for name, cfg in sims.items():
            print(f"    {name}: {cfg}")

    props = mappings.get("properties", {})
    print(f"  [fields]  ({len(props)} total)")
    for fname, fdef in sorted(props.items()):
        ftype = fdef.get("type", "?")
        extra = ""
        if "similarity" in fdef:
            extra = f"  sim={fdef['similarity']}"
        if "dimension" in fdef:
            extra = f"  dim={fdef['dimension']}"
        print(f"    {fname}: {ftype}{extra}")




def _pass(msg: str) -> None:
    print(f"  [ok]  {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    sys.exit(1)


def _assert(condition: bool, msg: str) -> None:
    if condition:
        _pass(msg)
    else:
        _fail(msg)


def _section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Tiny fake corpus: 10 docs
# ---------------------------------------------------------------------------

MINI_CORPUS = [
    {"id": f"TEST_{i:04d}", "contents": f"Abstract number {i}. This is a synthetic test document about topic {i}."}
    for i in range(10)
]

# 4-dim fake embeddings (unit-normalised to simulate real L2-norm=1 vecs)
# Using dim=4 keeps the test fast and cheap (no real model loaded).
EMB_DIM    = 4
MINI_EMB_A = np.random.randn(10, EMB_DIM).astype(np.float32)
MINI_EMB_A = (MINI_EMB_A / np.linalg.norm(MINI_EMB_A, axis=1, keepdims=True))  # L2-normalise

MINI_EMB_B = np.random.randn(10, EMB_DIM).astype(np.float32)
MINI_EMB_B = (MINI_EMB_B / np.linalg.norm(MINI_EMB_B, axis=1, keepdims=True))


# ---------------------------------------------------------------------------
# Test index settings: small / cheap
# ---------------------------------------------------------------------------

TEST_SETTINGS = IndexSettings(
    n_shards         = 1,
    n_replicas       = 0,
    refresh_interval = "1s",
    ef_search        = 10,
    ef_construct     = 16,
    hnsw_m           = 4,
)

BASELINE_ENCODERS = [("msmarco", "sentence-transformers/msmarco-distilbert-base-v2", EMB_DIM)]
EXTRA_ENCODERS    = [
    ("msmarco", "sentence-transformers/msmarco-distilbert-base-v2", EMB_DIM),
    ("testenc",  "some/test-encoder", EMB_DIM),
]


# ---------------------------------------------------------------------------
# Main test sequence
# ---------------------------------------------------------------------------

def run_tests(index_name: str) -> None:

    load_dotenv(_ROOT / ".env")

    # ---- 1. Connect --------------------------------------------------------
    _section("1 -- Connect to OpenSearch")
    try:
        client = get_client()
        _pass("client created")
    except Exception as e:
        _fail(f"get_client() raised: {e}")

    check_health(client)
    _pass("cluster healthy")

    # ---- 2. Clean slate: delete if exists ----------------------------------
    _section("2 -- Delete test index (clean slate)")
    delete_index(client, index_name)
    _assert(not client.indices.exists(index=index_name), "index does not exist after delete")

    # ---- 3. Build mapping (dry run) ----------------------------------------
    _section("3 -- build_index_mapping() -- dry run (no OpenSearch call)")

    mapping = build_index_mapping(
        bm25_k1_b_pairs = [(1.2, 0.75)],   # -> contents_bm25_k12_b075
        lmjm_lambdas    = [0.7],
        lmdir_mus       = [75],
        encoders        = BASELINE_ENCODERS,
        settings        = TEST_SETTINGS,
    )

    props = mapping["mappings"]["properties"]
    sims  = mapping["settings"]["similarity"]

    expected_fields = {
        "doc_id", "contents_bm25_k12_b075",
        "contents_lmjm_07", "contents_lmdir_75", "embedding_msmarco",
    }
    _assert(set(props.keys()) == expected_fields,
            f"mapping fields: {set(props.keys())}")
    _assert("bm25_k12_b075_similarity" in sims, "bm25_k12_b075 similarity present")
    _assert("lmjm_07_similarity" in sims,        "lmjm_07 similarity present")
    _assert("lmdir_75_similarity" in sims,        "lmdir_75 similarity present")

    knn_fdef = props["embedding_msmarco"]
    _assert(knn_fdef["type"] == "knn_vector",           "embedding_msmarco type = knn_vector")
    _assert(knn_fdef["dimension"] == EMB_DIM,            f"embedding_msmarco dim = {EMB_DIM}")
    _assert(knn_fdef["method"]["parameters"]["m"] == TEST_SETTINGS.hnsw_m, "hnsw_m correct")
    _assert(knn_fdef["method"]["parameters"]["ef_construction"] == TEST_SETTINGS.ef_construct,
            "ef_construct correct")

    # ---- 4. create_or_update_index() — Branch A: new index -----------------
    _section("4 -- create_or_update_index() -- Branch A: new index")

    create_or_update_index(
        client, index_name,
        bm25_k1_b_pairs = [(1.2, 0.75)],
        lmjm_lambdas    = [0.7],
        lmdir_mus       = [75],
        encoders        = BASELINE_ENCODERS,
        settings        = TEST_SETTINGS,
    )
    _assert(client.indices.exists(index=index_name), "index exists after create")

    live = get_live_fields(client, index_name)
    _assert(live == expected_fields, f"live fields after create: {live}")
    _pass("Branch A (new index) [ok]")

    # ---- 5. Index 10 docs (baseline: 1 KNN field) --------------------------
    _section("5 -- index_documents() -- 10 docs, baseline fields")

    index_documents(client, index_name, MINI_CORPUS, {"embedding_msmarco": MINI_EMB_A})
    client.indices.refresh(index=index_name)

    count = client.count(index=index_name).get("count", 0)
    _assert(count == 10, f"doc count after index = {count}")

    hit = client.get(index=index_name, id="TEST_0000")["_source"]
    _assert("doc_id"                  in hit, "doc_id field in source")
    _assert("contents_bm25_k12_b075"  in hit, "contents_bm25_k12_b075 field in source")
    _assert("contents_lmjm_07"        in hit, "contents_lmjm_07 field in source")
    _assert("contents_lmdir_75"       in hit, "contents_lmdir_75 field in source")
    _assert("embedding_msmarco"       in hit, "embedding_msmarco field in source")
    _assert(len(hit["embedding_msmarco"]) == EMB_DIM, f"embedding_msmarco has {EMB_DIM} dims")

    # ---- 6. index_documents() idempotency ----------------------------------
    _section("6 -- index_documents() idempotency -- same corpus + same fields")

    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        index_documents(client, index_name, MINI_CORPUS, {"embedding_msmarco": MINI_EMB_A})
    out = buf.getvalue()
    _assert("Skipping" in out, f"second call was skipped (output: {out.strip()!r})")

    count2 = client.count(index=index_name).get("count", 0)
    _assert(count2 == 10, f"doc count still 10 after idempotent call")

    # ---- 7. verify_mapping() -----------------------------------------------
    _section("7 -- verify_mapping() -- must not raise")
    verify_mapping(client, index_name)
    _pass("verify_mapping() ran without error")

    # ---- 8. create_or_update_index() idempotency — Branch B, nothing to add -
    _section("8 -- create_or_update_index() idempotency -- Branch B: nothing to add")

    buf_idem = io.StringIO()
    with contextlib.redirect_stdout(buf_idem):
        create_or_update_index(
            client, index_name,
            bm25_k1_b_pairs = [(1.2, 0.75)],
            lmjm_lambdas    = [0.7],
            lmdir_mus       = [75],
            encoders        = BASELINE_ENCODERS,
            settings        = TEST_SETTINGS,
        )
    out_idem = buf_idem.getvalue()
    _assert("already complete" in out_idem, "no-op message when nothing to add")
    live_idem = get_live_fields(client, index_name)
    _assert(live_idem == expected_fields, "fields unchanged after idempotent call")
    _pass("Branch B idempotency [ok]")

    # ---- 9. create_or_update_index() add-fields — Branch B -----------------
    _section("9 -- create_or_update_index() add-fields -- Branch B: extend existing index")

    # Add a new BM25 config, a new LM-JM lambda, a new LM-Dir mu, and a 2nd KNN field.
    # This exercises the close→put_settings→open + put_mapping path.
    create_or_update_index(
        client, index_name,
        bm25_k1_b_pairs = [(1.2, 0.75), (1.5, 1.0)],   # (1.5,1.0) is new
        lmjm_lambdas    = [0.7, 0.1],                   # 0.1 is new
        lmdir_mus       = [75, 2000],                    # 2000 is new
        encoders        = EXTRA_ENCODERS,                # testenc is new
        settings        = TEST_SETTINGS,
    )

    live_extended = get_live_fields(client, index_name)
    full_fields_expected = {
        "doc_id",
        "contents_bm25_k12_b075", "contents_bm25_k15_b10",
        "contents_lmjm_07",       "contents_lmjm_01",
        "contents_lmdir_75",      "contents_lmdir_2000",
        "embedding_msmarco",      "embedding_testenc",
    }
    _assert(live_extended == full_fields_expected,
            f"live fields after add-fields: {live_extended}")

    raw = client.indices.get_settings(index=index_name)
    actual_key = list(raw.keys())[0]
    live_sims  = raw[actual_key]["settings"]["index"].get("similarity", {})
    _assert("bm25_k12_b075_similarity" in live_sims, "bm25_k12_b075 similarity registered")
    _assert("bm25_k15_b10_similarity"  in live_sims, "bm25_k15_b10 similarity registered")
    _assert("lmjm_07_similarity"       in live_sims, "lmjm_07 similarity registered")
    _assert("lmjm_01_similarity"       in live_sims, "lmjm_01 similarity registered")
    _assert("lmdir_75_similarity"      in live_sims, "lmdir_75 similarity registered")
    _assert("lmdir_2000_similarity"    in live_sims, "lmdir_2000 similarity registered")
    _pass("Branch B add-fields [ok]")

    # ---- 10. Index 10 docs with 2 KNN fields --------------------------------
    _section("10 -- index_documents() -- 10 docs, 2 KNN fields")

    index_documents(
        client, index_name, MINI_CORPUS,
        {"embedding_msmarco": MINI_EMB_A, "embedding_testenc": MINI_EMB_B},
    )
    client.indices.refresh(index=index_name)

    count3 = client.count(index=index_name).get("count", 0)
    _assert(count3 == 10, f"doc count = {count3}")

    hit2 = client.get(index=index_name, id="TEST_0005")["_source"]
    for f in full_fields_expected:
        _assert(f in hit2, f"field '{f}' present in source")
    _assert(len(hit2["embedding_msmarco"]) == EMB_DIM, "embedding_msmarco dim ok")
    _assert(len(hit2["embedding_testenc"]) == EMB_DIM, "embedding_testenc dim ok")

    # ---- 11. New-field detection: index_documents must re-index -------------
    _section("11 -- new-field detection -- index_documents re-indexes on new KNN field")

    client.indices.put_mapping(
        index = index_name,
        body  = {
            "properties": {
                "embedding_extra": {
                    "type":      "knn_vector",
                    "dimension": EMB_DIM,
                    "method": {
                        "name":       "hnsw",
                        "space_type": "innerproduct",
                        "engine":     "faiss",
                        "parameters": {"ef_construction": 16, "m": 4},
                    },
                }
            }
        },
    )
    live_after_put = get_live_fields(client, index_name)
    _assert("embedding_extra" in live_after_put, "embedding_extra field added via put_mapping")

    MINI_EMB_EXTRA = np.random.randn(10, EMB_DIM).astype(np.float32)
    MINI_EMB_EXTRA = MINI_EMB_EXTRA / np.linalg.norm(MINI_EMB_EXTRA, axis=1, keepdims=True)

    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2):
        index_documents(
            client, index_name, MINI_CORPUS,
            {
                "embedding_msmarco": MINI_EMB_A,
                "embedding_testenc": MINI_EMB_B,
                "embedding_extra":   MINI_EMB_EXTRA,
            },
        )
    out2 = buf2.getvalue()
    _assert("Skipping" not in out2, "re-index triggered when new KNN field added")
    _assert("re-index" in out2.lower() or "new field" in out2.lower(),
            "log mentions new field or re-index")

    client.indices.refresh(index=index_name)
    hit3 = client.get(index=index_name, id="TEST_0000")["_source"]
    _assert("embedding_extra" in hit3,            "embedding_extra populated after re-index")
    _assert(len(hit3["embedding_extra"]) == EMB_DIM, "embedding_extra dim ok")

    # ---- 12. Shape mismatch guard ------------------------------------------
    _section("12 -- embeddings_map shape mismatch must raise ValueError")
    bad_emb = np.zeros((5, EMB_DIM), dtype=np.float32)   # 5 rows, corpus has 10
    raised = False
    try:
        index_documents(client, index_name, MINI_CORPUS, {"embedding_msmarco": bad_emb})
    except ValueError as e:
        raised = True
        _pass(f"ValueError raised as expected: {e}")
    _assert(raised, "ValueError raised for shape mismatch")

    # ---- 13. KNN dim mismatch warning in create_or_update_index ------------
    _section("13 -- KNN dim mismatch warning -- create_or_update_index warns, does not crash")

    # Request a KNN encoder with a DIFFERENT dim than what's in the live index (EMB_DIM).
    WRONG_DIM = EMB_DIM + 4
    buf3 = io.StringIO()
    with contextlib.redirect_stdout(buf3):
        create_or_update_index(
            client, index_name,
            bm25_k1_b_pairs = [(1.2, 0.75)],
            lmjm_lambdas    = [0.7],
            lmdir_mus       = [75],
            encoders        = [("msmarco", "some/model", WRONG_DIM)],
            settings        = TEST_SETTINGS,
        )
    out3 = buf3.getvalue()
    _assert("dim mismatch" in out3.lower() or "mismatch" in out3.lower(),
            f"dim mismatch warning printed (got: {out3.strip()!r})")
    _assert(client.indices.exists(index=index_name), "index still exists after mismatch warning")
    _pass("KNN dim mismatch warning [ok]")

    # ---- 14. Clean up -------------------------------------------------------
    _section("14 -- cleanup: delete test index")
    delete_index(client, index_name)
    _assert(not client.indices.exists(index=index_name), "index gone after final delete")

    # ---- Done ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  ALL TESTS PASSED [ok]")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    load_dotenv(_ROOT / ".env")
    _index = os.getenv("OPENSEARCH_INDEX", "")
    if not _index:
        print("ERROR: OPENSEARCH_INDEX not set in .env")
        sys.exit(1)

    # use a dedicated test index so we never touch the real production index
    test_index = f"{_index}_test"
    print(f"Test index : {test_index}")
    run_tests(test_index)
