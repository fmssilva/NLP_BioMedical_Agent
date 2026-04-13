from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.data.loader import load_corpus
from src.embeddings.encoder import Encoder, POOLING_MEAN, POOLING_CLS


######################################################################
## Corpus encoder 
######################################################################

# MedCPT uses an asymmetric dual-encoder design:
#   - ncbi/MedCPT-Article-Encoder  → corpus documents  (CLS pooling, max_length=512)
#   - ncbi/MedCPT-Query-Encoder    → search queries     (CLS pooling, max_length=64)
# The ENCODER_MED_CPT tuple in constants still points to the Query-Encoder because
# that alias is also used at query time (KNNRetriever / MedCPTKNNRetriever).
# create_embeddings() detects the "medcpt" alias and substitutes the Article-Encoder
# + CLS pooling for the offline corpus encoding step.
_MEDCPT_ALIAS          = "medcpt"
_MEDCPT_ARTICLE_MODEL  = "ncbi/MedCPT-Article-Encoder"

def encode_corpus(encoder: Encoder, corpus: list[dict], batch_size: int = 32) -> np.ndarray:
    """
    Encode all corpus documents in batches.
    """
    # extract text, encode in batches, stack into (N, dim) array
    texts = [doc["contents"] for doc in corpus]
    all_vecs = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding corpus", unit="batch"):
        batch = texts[start: start + batch_size]
        all_vecs.append(encoder.encode(batch, batch_size=batch_size))
    return np.vstack(all_vecs)


def save_embeddings(vectors: np.ndarray, path: str | Path) -> None:
    """
    Save embedding matrix to disk as .npy.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors)
    print(f"[corpus_encoder] Saved {vectors.shape} embeddings -> {path}")


def load_embeddings(path: str | Path) -> np.ndarray:
    """
    Load embedding matrix from .npy 
    """
    vectors = np.load(path)
    print(f"[corpus_encoder] Loaded embeddings {vectors.shape} <- {path}")
    return vectors


def create_embeddings(
    corpus:       list[dict],
    models:       list[tuple[str, str, int]],
    output_dir:   str | Path,
    batch_size:   int = 32,
    force:        bool = False,
    pooling_mode: str = POOLING_MEAN,
) -> list[tuple[str, str, np.ndarray]]:
    """
    Encode corpus with each model in ``models``.
    ``models`` is a list of (alias, hf_model_id, dim) tuples — e.g. ENCODERS_LIST.
    Returns list of (alias, model_name, vectors)
    Each model gets its own cache file: output_dir/{alias}.npy
    - loads the .npy if it exists and force=False, encodes + saves otherwise
    - slices vectors to len(corpus) if a cached file has more rows (CORPUS_SIZE testing mode)
    - logs shape + L2 norm sample after each model so encoding is easy to verify
    """
    assert len(models) > 0, "models list must not be empty"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # returns list of (alias, model_name, vectors) — one entry per model, same order as input
    results: list[tuple[str, str, np.ndarray]] = []

    for alias, model_name, _dim in models:
        npy_path = output_dir / f"{alias}.npy"

        needs_encode = force or not npy_path.exists()

        if not needs_encode:
            # Load cache and check it covers the full corpus
            vectors = load_embeddings(npy_path)
            if vectors.shape[0] < len(corpus):
                # Cache was built on a smaller corpus subset (e.g. CORPUS_SIZE=10 smoke-test).
                # Must re-encode the full corpus — using the stale cache would silently
                # give wrong shapes downstream.
                print(
                    f"[corpus_encoder] '{alias}': cached file has only {vectors.shape[0]} rows "
                    f"but corpus has {len(corpus)} docs — re-encoding full corpus."
                )
                needs_encode = True

        if needs_encode:
            reason = "FORCE_REENCODE" if force else "no cache / stale cache"
            # MedCPT uses the Article-Encoder for corpus docs (CLS pooling, max_length=512).
            # All other models use the default pooling_mode passed to create_embeddings().
            if alias == _MEDCPT_ALIAS:
                actual_model   = _MEDCPT_ARTICLE_MODEL
                actual_pooling = POOLING_CLS
                print(f"[corpus_encoder] '{alias}': overriding model -> '{actual_model}' "
                      f"(Article-Encoder) with CLS pooling ({reason})")
            else:
                actual_model   = model_name
                actual_pooling = pooling_mode
                print(f"[corpus_encoder] '{alias}': encoding {len(corpus)} docs with '{actual_model}' ({reason})")
            encoder = Encoder(actual_model, pooling_mode=actual_pooling)
            vectors = encode_corpus(encoder, corpus, batch_size=batch_size)
            save_embeddings(vectors, npy_path)

        # Trim if cache was built on a larger corpus and now we are testing with smaller corpus 
        if vectors.shape[0] > len(corpus):
            print(f"[corpus_encoder] '{alias}': trimming cache {vectors.shape[0]} → {len(corpus)} rows")
            vectors = vectors[:len(corpus)]

        # sanity: shape + L2 norms (all should be ~1.0 for L2-normalised encoders)
        norms = np.linalg.norm(vectors[:5], axis=1).round(4).tolist()
        print(f"[corpus_encoder] '{alias}': shape={vectors.shape}  L2 norms sample={norms}")

        results.append((alias, model_name, vectors))

    return results
