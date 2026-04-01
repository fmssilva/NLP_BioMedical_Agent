"""
src/embeddings/__encoder_test.py

Comprehensive tests for encoder.py and corpus_encoder.py using CORPUS_SIZE=10.

Tests:
  encoder.py:
    - Encoder loads, encode() shape (N, 768), L2 norms ~1.0
    - encode_single shape (768,), consistent with encode([text])[0]
    - Cosine sim of same text == 1.0; different texts < 0.99
    - resolve_model_name: aliases -> full IDs, full IDs pass through

  corpus_encoder.py — single model:
    - encode_corpus shape + L2 norms
    - save/load roundtrip
    - create_embeddings returns list[(alias, ndarray)] with 1 entry
    - Cache hit: second call loads file, no re-encode
    - force=True: re-encodes even if file exists, same vectors
    - alias used as file name (not the full model ID)

  corpus_encoder.py — multi-model:
    - 2 aliases -> 2 entries, correct order preserved
    - Each alias -> its own .npy file (name-file alignment)
    - Same model under different aliases -> same vectors (consistency)
    - Each alias file holds only its own vectors (no cross-contamination)
    - Partial cache: existing alias loaded, missing alias encoded fresh
    - All returned arrays have correct shapes and L2 norms
    - embeddings_map built from results has correct field names for index_documents

Run with:
    cd C:\\Users\\franc\\Desktop\\NLP_Biomedical_Agent
    C:/Users/franc/anaconda3/envs/cnn/python.exe -m src.embeddings.__encoder_test
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.loader import load_corpus
from src.embeddings.encoder import Encoder
from src.embeddings.corpus_encoder import (
    encode_corpus, save_embeddings, load_embeddings, create_embeddings
)

CORPUS_SIZE  = 10
CORPUS_PATH  = _ROOT / "data" / "filtered_pubmed_abstracts.txt"
EXPECTED_DIM = 768

# alias -> full HF model ID  (mirrors ENCODER_MAP in the notebook constants)
MODEL_MAP = {
    "msmarco":  "sentence-transformers/msmarco-distilbert-base-v2",
    "medcpt":   "ncbi/MedCPT-Query-Encoder",
    "multi-qa": "sentence-transformers/multi-qa-mpnet-base-cos-v1",
}
def resolve_model_name(alias: str) -> str:
    return MODEL_MAP.get(alias, alias)
    

# load corpus + encoder once; all tests reuse them to stay fast
_corpus10 = None
_enc      = None

def _get_corpus():
    global _corpus10
    if _corpus10 is None:
        _corpus10 = load_corpus(CORPUS_PATH, size=CORPUS_SIZE)
    return _corpus10

def _get_encoder():
    global _enc
    if _enc is None:
        _enc = Encoder()   # msmarco by default
    return _enc


# ── resolve_model_name ────────────────────────────────────────────────────

def test_resolve_aliases():
    assert resolve_model_name("msmarco")  == "sentence-transformers/msmarco-distilbert-base-v2"
    assert resolve_model_name("medcpt")   == "ncbi/MedCPT-Query-Encoder"
    assert resolve_model_name("multi-qa") == "sentence-transformers/multi-qa-mpnet-base-cos-v1"
    print("  [ok]  resolve_model_name: all 3 aliases map to correct full IDs")

def test_resolve_passthrough():
    full = "sentence-transformers/msmarco-distilbert-base-v2"
    assert resolve_model_name(full) == full
    assert resolve_model_name("some/unknown") == "some/unknown"
    print("  [ok]  resolve_model_name: full IDs / unknowns pass through unchanged")


# ── Encoder class ─────────────────────────────────────────────────────────

def test_encoder_loads():
    enc = _get_encoder()
    assert enc.model is not None and enc.tokenizer is not None
    print(f"  [ok]  Encoder loaded on device={enc.device}")

def test_encode_shape():
    texts = [d["contents"] for d in _get_corpus()]
    vecs  = _get_encoder().encode(texts, batch_size=CORPUS_SIZE)
    assert vecs.shape == (CORPUS_SIZE, EXPECTED_DIM), f"shape wrong: {vecs.shape}"
    print(f"  [ok]  encode({CORPUS_SIZE} texts) -> {vecs.shape}")

def test_encode_l2_norms():
    texts = [d["contents"] for d in _get_corpus()]
    vecs  = _get_encoder().encode(texts, batch_size=CORPUS_SIZE)
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), f"norms not ~1.0: {norms}"
    print(f"  [ok]  all L2 norms ~1.0 (min={norms.min():.6f}  max={norms.max():.6f})")

def test_encode_single():
    text = _get_corpus()[0]["contents"]
    vec  = _get_encoder().encode_single(text)
    assert vec.shape == (EXPECTED_DIM,), f"shape wrong: {vec.shape}"
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-5
    v_batch = _get_encoder().encode([text])[0]
    assert np.allclose(vec, v_batch, atol=1e-6), "encode_single != encode([text])[0]"
    print(f"  [ok]  encode_single: shape {vec.shape}, norm ~1.0, consistent with encode()")

def test_cosine_self():
    v = _get_encoder().encode_single("obstructive sleep apnea treatment CPAP")
    assert abs(float(np.dot(v, v)) - 1.0) < 1e-5
    print(f"  [ok]  cosine(text, text) = 1.0")

def test_cosine_different():
    v1 = _get_encoder().encode_single("sleep apnea CPAP")
    v2 = _get_encoder().encode_single("hepatitis B antiviral ribavirin")
    assert float(np.dot(v1, v2)) < 0.99
    print(f"  [ok]  different texts: cos sim = {float(np.dot(v1,v2)):.4f} (< 0.99)")


# ── encode_corpus + save/load ─────────────────────────────────────────────

def test_encode_corpus_shape():
    vecs = encode_corpus(_get_encoder(), _get_corpus(), batch_size=CORPUS_SIZE)
    assert vecs.shape == (CORPUS_SIZE, EXPECTED_DIM)
    print(f"  [ok]  encode_corpus({CORPUS_SIZE} docs) -> {vecs.shape}")

def test_encode_corpus_l2():
    vecs  = encode_corpus(_get_encoder(), _get_corpus(), batch_size=CORPUS_SIZE)
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    print(f"  [ok]  encode_corpus: all L2 norms ~1.0")

def test_save_load_roundtrip():
    vecs = encode_corpus(_get_encoder(), _get_corpus(), batch_size=CORPUS_SIZE)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.npy"
        save_embeddings(vecs, p)
        assert p.exists()
        loaded = load_embeddings(p)
        assert loaded.shape == vecs.shape
        assert np.allclose(vecs, loaded, atol=1e-7)
    print(f"  [ok]  save/load roundtrip: shapes match, values identical")


# ── create_embeddings — single model ─────────────────────────────────────

def test_single_model_returns_list_of_one():
    with tempfile.TemporaryDirectory() as tmp:
        results = create_embeddings(_get_corpus(), ["msmarco"], MODEL_MAP, tmp, batch_size=CORPUS_SIZE)
        assert isinstance(results, list) and len(results) == 1
        alias, model_name, vecs = results[0]
        assert alias == "msmarco",                       f"alias wrong: {alias}"
        assert model_name == MODEL_MAP["msmarco"],       f"model_name wrong: {model_name}"
        assert vecs.shape == (CORPUS_SIZE, EXPECTED_DIM)
        assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)
    print(f"  [ok]  single model: list[(alias, model_name, ndarray)], 1 entry, correct alias+shape+norms")

def test_single_model_creates_npy_file():
    # file must be named {alias}.npy, NOT derived from the full model ID
    with tempfile.TemporaryDirectory() as tmp:
        create_embeddings(_get_corpus(), ["msmarco"], MODEL_MAP, tmp, batch_size=CORPUS_SIZE)
        assert (Path(tmp) / "msmarco.npy").exists(), "msmarco.npy not created"
    print(f"  [ok]  single model: cache file named 'msmarco.npy' (alias, not full model ID)")

def test_single_model_cache_hit():
    with tempfile.TemporaryDirectory() as tmp:
        r1 = create_embeddings(_get_corpus(), ["msmarco"], MODEL_MAP, tmp, batch_size=CORPUS_SIZE)
        _, _, v1 = r1[0]
        r2 = create_embeddings(_get_corpus(), ["msmarco"], MODEL_MAP, tmp, batch_size=CORPUS_SIZE)
        _, _, v2 = r2[0]
        assert np.allclose(v1, v2, atol=1e-7), "cached vectors differ from original"
    print(f"  [ok]  single model: second call loads cache, vectors identical")

def test_single_model_force_reencode():
    with tempfile.TemporaryDirectory() as tmp:
        r1 = create_embeddings(_get_corpus(), ["msmarco"], MODEL_MAP, tmp, batch_size=CORPUS_SIZE)
        _, _, v1 = r1[0]
        # overwrite with zeros to confirm force=True replaces it
        np.save(Path(tmp) / "msmarco.npy", np.zeros_like(v1))
        r2 = create_embeddings(_get_corpus(), ["msmarco"], MODEL_MAP, tmp, batch_size=CORPUS_SIZE, force=True)
        _, _, v2 = r2[0]
        assert not np.allclose(v2, 0.0),       "force=True still returned zeros — encoder did not run"
        assert np.allclose(v1, v2, atol=1e-5), "force re-encode produced different vectors"
    print(f"  [ok]  single model: force=True re-encodes and overwrites zeros")


# ── create_embeddings — multi-model ──────────────────────────────────────
# use two aliases that both resolve to msmarco to avoid downloading medcpt in CI

def test_multi_model_returns_correct_count():
    fake_map = {
        "alias_A": "sentence-transformers/msmarco-distilbert-base-v2",
        "alias_B": "sentence-transformers/msmarco-distilbert-base-v2",
    }
    with tempfile.TemporaryDirectory() as tmp:
        results = create_embeddings(_get_corpus(), ["alias_A", "alias_B"], fake_map, tmp, batch_size=CORPUS_SIZE)
        assert len(results) == 2, f"expected 2 results, got {len(results)}"
    print(f"  [ok]  multi-model: 2 aliases -> 2 entries in results list")

def test_multi_model_order_preserved():
    # returned list must be in same order as input aliases list
    fake_map = {
        "first":  "sentence-transformers/msmarco-distilbert-base-v2",
        "second": "sentence-transformers/msmarco-distilbert-base-v2",
    }
    with tempfile.TemporaryDirectory() as tmp:
        results = create_embeddings(_get_corpus(), ["first", "second"], fake_map, tmp, batch_size=CORPUS_SIZE)
        assert results[0][0] == "first",  f"expected 'first', got '{results[0][0]}'"
        assert results[1][0] == "second", f"expected 'second', got '{results[1][0]}'"
    print(f"  [ok]  multi-model: return order matches input order ['first','second']")

def test_multi_model_file_names():
    # each alias -> its own {alias}.npy (not overwriting each other)
    fake_map = {
        "alpha": "sentence-transformers/msmarco-distilbert-base-v2",
        "beta":  "sentence-transformers/msmarco-distilbert-base-v2",
    }
    with tempfile.TemporaryDirectory() as tmp:
        create_embeddings(_get_corpus(), ["alpha", "beta"], fake_map, tmp, batch_size=CORPUS_SIZE)
        assert (Path(tmp) / "alpha.npy").exists(), "alpha.npy not created"
        assert (Path(tmp) / "beta.npy").exists(),  "beta.npy not created"
    print(f"  [ok]  multi-model: each alias has its own .npy file (alpha.npy, beta.npy)")

def test_multi_model_no_cross_contamination():
    # each result entry must match its own .npy, not the other model's file
    fake_map = {
        "aaa": "sentence-transformers/msmarco-distilbert-base-v2",
        "bbb": "sentence-transformers/msmarco-distilbert-base-v2",
    }
    with tempfile.TemporaryDirectory() as tmp:
        results       = create_embeddings(_get_corpus(), ["aaa", "bbb"], fake_map, tmp, batch_size=CORPUS_SIZE)
        _, _, v_aaa   = results[0]
        _, _, v_bbb   = results[1]
        loaded_a = np.load(Path(tmp) / "aaa.npy")
        loaded_b = np.load(Path(tmp) / "bbb.npy")
        # each result entry matches its file
        assert np.allclose(v_aaa, loaded_a, atol=1e-7), "aaa result does not match aaa.npy"
        assert np.allclose(v_bbb, loaded_b, atol=1e-7), "bbb result does not match bbb.npy"
        # same model -> same content in both files
        assert np.allclose(loaded_a, loaded_b, atol=1e-5), "same model but files differ"
    print(f"  [ok]  multi-model: each alias file holds correct vectors, no cross-contamination")

def test_multi_model_partial_cache():
    # pre-populate alias_A; second call should reuse it and only encode alias_B
    fake_map = {
        "alias_A": "sentence-transformers/msmarco-distilbert-base-v2",
        "alias_B": "sentence-transformers/msmarco-distilbert-base-v2",
    }
    with tempfile.TemporaryDirectory() as tmp:
        # encode A only
        r_a = create_embeddings(_get_corpus(), ["alias_A"], fake_map, tmp, batch_size=CORPUS_SIZE)
        v_a_original = r_a[0][2].copy()   # index 2 = vectors
        mtime_a = (Path(tmp) / "alias_A.npy").stat().st_mtime
        # now encode both — A must be loaded from cache, B encoded fresh
        results = create_embeddings(_get_corpus(), ["alias_A", "alias_B"], fake_map, tmp, batch_size=CORPUS_SIZE)
        assert len(results) == 2
        _, _, v_a_from_cache = results[0]
        assert np.allclose(v_a_original, v_a_from_cache, atol=1e-7), "alias_A cache was not reused"
        # A's file mtime should not have changed (no re-write)
        mtime_a2 = (Path(tmp) / "alias_A.npy").stat().st_mtime
        assert mtime_a == mtime_a2, "alias_A.npy was re-written even though it existed"
        assert (Path(tmp) / "alias_B.npy").exists(), "alias_B.npy was not created"
    print(f"  [ok]  multi-model: partial cache — existing alias loaded, missing one encoded")

def test_multi_model_shapes_and_norms():
    fake_map = {
        "x": "sentence-transformers/msmarco-distilbert-base-v2",
        "y": "sentence-transformers/msmarco-distilbert-base-v2",
    }
    with tempfile.TemporaryDirectory() as tmp:
        results = create_embeddings(_get_corpus(), ["x", "y"], fake_map, tmp, batch_size=CORPUS_SIZE)
        for alias, _model, vecs in results:
            assert vecs.shape == (CORPUS_SIZE, EXPECTED_DIM), f"[{alias}] shape wrong: {vecs.shape}"
            norms = np.linalg.norm(vecs, axis=1)
            assert np.allclose(norms, 1.0, atol=1e-5), f"[{alias}] norms not ~1.0"
    print(f"  [ok]  multi-model: all arrays shape ({CORPUS_SIZE},{EXPECTED_DIM}), norms ~1.0")


# ── field-name alignment: alias -> OpenSearch field name ─────────────────

def test_embeddings_field_names():
    """
    Verify document_indexer's alias->field convention:
      "msmarco" -> "embedding"   (baseline, no suffix)
      "X"       -> "embedding_X"
    This is the same mapping that index_documents() does internally.
    """
    fake_map = {
        "msmarco": "sentence-transformers/msmarco-distilbert-base-v2",
        "testenc": "sentence-transformers/msmarco-distilbert-base-v2",
    }
    with tempfile.TemporaryDirectory() as tmp:
        results = create_embeddings(_get_corpus(), ["msmarco", "testenc"], fake_map, tmp, batch_size=CORPUS_SIZE)

    # replicate the mapping logic from document_indexer.index_documents()
    embeddings_map = {
        ("embedding" if alias == "msmarco" else f"embedding_{alias}"): vecs
        for alias, _model, vecs in results
    }

    assert "embedding"         in embeddings_map, "msmarco must map to 'embedding'"
    assert "embedding_testenc" in embeddings_map, "testenc must map to 'embedding_testenc'"
    assert len(embeddings_map) == 2
    for field, vecs in embeddings_map.items():
        assert vecs.shape == (CORPUS_SIZE, EXPECTED_DIM), f"[{field}] shape wrong: {vecs.shape}"

    print(f"  [ok]  field names: {list(embeddings_map.keys())} match index convention")


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("encoder + corpus_encoder tests  (CORPUS_SIZE=10)")
    print("=" * 60)

    print("\n-- resolve_model_name --")
    test_resolve_aliases()
    test_resolve_passthrough()

    print("\n-- Encoder (model loaded once, reused across tests) --")
    test_encoder_loads()
    test_encode_shape()
    test_encode_l2_norms()
    test_encode_single()
    test_cosine_self()
    test_cosine_different()

    print("\n-- encode_corpus + save/load --")
    test_encode_corpus_shape()
    test_encode_corpus_l2()
    test_save_load_roundtrip()

    print("\n-- create_embeddings: single model --")
    test_single_model_returns_list_of_one()
    test_single_model_creates_npy_file()
    test_single_model_cache_hit()
    test_single_model_force_reencode()

    print("\n-- create_embeddings: multi-model --")
    test_multi_model_returns_correct_count()
    test_multi_model_order_preserved()
    test_multi_model_file_names()
    test_multi_model_no_cross_contamination()
    test_multi_model_partial_cache()
    test_multi_model_shapes_and_norms()

    print("\n-- embeddings_map field-name alignment --")
    test_embeddings_field_names()

    print("\n" + "=" * 60)
    print("All encoder + corpus_encoder tests passed.")
    print("=" * 60)
