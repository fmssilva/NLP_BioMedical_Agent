"""
src/embeddings/corpus_encoder.py

Encode corpus documents offline, one .npy per encoder model.

Public API:
    encode_corpus(encoder, corpus, batch_size)          -> np.ndarray (N, dim)
    save_embeddings(vectors, path)                      -> None
    load_embeddings(path)                               -> np.ndarray
    create_embeddings(corpus, models, model_map,
                      output_dir, batch_size, force)    -> list[(alias, model_name, ndarray)]

The .npy files are gitignored (~25 MB each). Re-generate any time with __main__.
"""

from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.data.loader import load_corpus
from src.embeddings.encoder import Encoder


# Encode all corpus documents in batches.
def encode_corpus(encoder: Encoder, corpus: list[dict], batch_size: int = 32) -> np.ndarray:
    # extract text, encode in batches, stack into (N, dim) array
    texts = [doc["contents"] for doc in corpus]
    all_vecs = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding corpus", unit="batch"):
        batch = texts[start: start + batch_size]
        all_vecs.append(encoder.encode(batch, batch_size=batch_size))
    return np.vstack(all_vecs)


# Save embedding matrix to disk as .npy.
def save_embeddings(vectors: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors)
    print(f"[corpus_encoder] Saved {vectors.shape} embeddings -> {path}")


# Load embedding matrix from .npy — fast, no re-encoding needed.
def load_embeddings(path: str | Path) -> np.ndarray:
    vectors = np.load(path)
    print(f"[corpus_encoder] Loaded embeddings {vectors.shape} <- {path}")
    return vectors


def create_embeddings(
    corpus:     list[dict],
    models:     list[str],
    model_map:  dict[str, str],
    output_dir: str | Path,
    batch_size: int = 32,
    force:      bool = False,
) -> list[tuple[str, str, np.ndarray]]:
    """
    Encode corpus with each model in ``models``.

    Returns list of (alias, model_name, vectors) — one entry per model, same order as input.
    Each model gets its own cache file: output_dir/{alias}.npy
    - loads the .npy if it exists and force=False, encodes + saves otherwise
    - slices vectors to len(corpus) if a cached file has more rows (CORPUS_SIZE testing mode)
    - logs shape + L2 norm sample after each model so encoding is easy to verify
    """
    assert len(models) > 0, "models list must not be empty"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[tuple[str, str, np.ndarray]] = []

    for alias in models:
        # resolve alias -> full HF model ID (pass-through if alias not in map)
        model_name = model_map.get(alias, alias)
        npy_path   = output_dir / f"{alias}.npy"

        if npy_path.exists() and not force:
            print(f"[corpus_encoder] '{alias}': loading cached embeddings from {npy_path}")
            vectors = load_embeddings(npy_path)
        else:
            if force and npy_path.exists():
                print(f"[corpus_encoder] '{alias}': FORCE_REENCODE — re-encoding with '{model_name}'")
            else:
                print(f"[corpus_encoder] '{alias}': encoding {len(corpus)} docs with '{model_name}'")
            encoder = Encoder(model_name=model_name)
            vectors = encode_corpus(encoder, corpus, batch_size=batch_size)
            save_embeddings(vectors, npy_path)

        # slice to corpus length — cached .npy may have more rows (e.g. full 4194 vs CORPUS_SIZE=10)
        if vectors.shape[0] != len(corpus):
            print(f"[corpus_encoder] '{alias}': slicing {vectors.shape[0]} -> {len(corpus)} rows")
            vectors = vectors[:len(corpus)]

        # sanity: shape + L2 norms (all should be ~1.0 for L2-normalised encoders)
        norms = np.linalg.norm(vectors[:5], axis=1).round(4).tolist()
        print(f"[corpus_encoder] '{alias}': shape={vectors.shape}  L2 norms sample={norms}")

        assert vectors.shape[0] == len(corpus), (
            f"[corpus_encoder] '{alias}': vector count {vectors.shape[0]} != corpus {len(corpus)}"
        )
        results.append((alias, model_name, vectors))

    return results


##############################################################################
##                      LOCAL TEST                                          ##
##############################################################################
if __name__ == "__main__":
    import sys, tempfile

    print("=" * 60)
    print("corpus_encoder.py  — self-test  (CORPUS_SIZE=10)")
    print("=" * 60)

    root        = Path(__file__).resolve().parents[2]
    corpus_path = root / "data" / "filtered_pubmed_abstracts.txt"

    MODEL_MAP = {
        "msmarco":  "sentence-transformers/msmarco-distilbert-base-v2",
        "medcpt":   "ncbi/MedCPT-Query-Encoder",
        "multi-qa": "sentence-transformers/multi-qa-mpnet-base-cos-v1",
    }

    # 1. load 10 docs for fast smoke tests
    print("\n[1/4] Loading 10-doc corpus ...")
    corpus10 = load_corpus(corpus_path, size=10)
    print(f"      {len(corpus10)} docs loaded")

    # 2. encode_corpus low-level test
    print("\n[2/4] Smoke: encode_corpus (msmarco, 10 docs) ...")
    enc = Encoder()
    vecs = encode_corpus(enc, corpus10, batch_size=10)
    assert vecs.shape == (10, 768), f"shape {vecs.shape}"
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), f"norms: {norms}"
    print(f"      shape={vecs.shape}  norms min={norms.min():.6f} max={norms.max():.6f}  OK")

    # 3. create_embeddings — single model, cache, force
    print("\n[3/4] create_embeddings — single model (msmarco) ...")
    with tempfile.TemporaryDirectory() as tmpdir:
        # first call: encodes + saves
        r1 = create_embeddings(corpus10, ["msmarco"], MODEL_MAP, tmpdir, batch_size=10)
        assert len(r1) == 1,                         f"expected 1 result, got {len(r1)}"
        alias1, model1, v1 = r1[0]
        assert alias1 == "msmarco",                  f"alias wrong: {alias1}"
        assert model1 == MODEL_MAP["msmarco"],        f"model_name wrong: {model1}"
        assert v1.shape == (10, 768),                f"shape wrong: {v1.shape}"
        assert (Path(tmpdir) / "msmarco.npy").exists(), "cache file not created"
        print(f"      first call: alias={alias1}  shape={v1.shape}  [ok]")

        # second call: loads from cache (no re-encode)
        r2 = create_embeddings(corpus10, ["msmarco"], MODEL_MAP, tmpdir, batch_size=10)
        alias2, _, v2 = r2[0]
        assert np.allclose(v1, v2, atol=1e-7), "cached vectors differ from original"
        print(f"      cache hit:  alias={alias2}  shapes match  [ok]")

        # force=True: re-encodes even though file exists
        r3 = create_embeddings(corpus10, ["msmarco"], MODEL_MAP, tmpdir, batch_size=10, force=True)
        _, _, v3 = r3[0]
        assert np.allclose(v1, v3, atol=1e-5), "force re-encode produced different vectors"
        print(f"      force=True: re-encoded, vectors match  [ok]")

    # 4. create_embeddings — order/name alignment for multi-model list
    # (use msmarco twice under different aliases to avoid loading medcpt/multi-qa in self-test)
    print("\n[4/4] create_embeddings — multi-alias, name-vector alignment ...")
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_map = {
            "alias_A": "sentence-transformers/msmarco-distilbert-base-v2",
            "alias_B": "sentence-transformers/msmarco-distilbert-base-v2",
        }
        results = create_embeddings(corpus10, ["alias_A", "alias_B"], fake_map, tmpdir, batch_size=10)
        assert len(results) == 2, f"expected 2, got {len(results)}"
        a_name, a_model, a_vecs = results[0]
        b_name, b_model, b_vecs = results[1]
        assert a_name == "alias_A", f"first alias wrong: {a_name}"
        assert b_name == "alias_B", f"second alias wrong: {b_name}"
        assert a_vecs.shape == (10, 768), f"A shape wrong: {a_vecs.shape}"
        assert b_vecs.shape == (10, 768), f"B shape wrong: {b_vecs.shape}"
        # both use same model -> should be equal
        assert np.allclose(a_vecs, b_vecs, atol=1e-5), "same model, different aliases -> vecs should match"
        # files: alias_A.npy and alias_B.npy
        assert (Path(tmpdir) / "alias_A.npy").exists(), "alias_A.npy not created"
        assert (Path(tmpdir) / "alias_B.npy").exists(), "alias_B.npy not created"
        print(f"      order: [{a_name}, {b_name}]  shapes: {a_vecs.shape}  name-file alignment  [ok]")

    print("\n" + "=" * 60)
    print("corpus_encoder.py  —  all tests passed")
    print("=" * 60)

