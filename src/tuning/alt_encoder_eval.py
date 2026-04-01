"""
src/tuning/alt_encoder_eval.py

Compare alternative dense encoders on the train set.

Current baseline: msmarco-distilbert-base-v2 (KNN MAP=0.4337 on train)
Problem: This is a web search model — domain mismatch for PubMed biomedical text.
Candidates:
  - ncbi/MedCPT-Query-Encoder + ncbi/MedCPT-Article-Encoder  (biomedical, highest priority)
  - sentence-transformers/multi-qa-mpnet-base-cos-v1          (general QA, high quality)

Evaluation approach: PURE PYTHON — no OpenSearch needed.
  For each encoder:
    1. Encode all 4194 corpus docs with the document encoder
    2. Encode each train query with the query encoder
    3. Compute exact cosine similarity (brute-force dot product on L2-normalised vectors)
    4. Rank documents by similarity score
    5. Compute MAP/MRR/P@10 directly

This is 100% accurate (no HNSW approximation) and runs entirely on CPU.
Comparing encoders this way tells us exactly which one is better without
touching the OpenSearch index. Only if an encoder wins do we add a KNN field.

Usage:
    python -m src.tuning.alt_encoder_eval           # run comparison, save CSV
    python -m src.tuning.alt_encoder_eval --show    # print existing CSV
    python -m src.tuning.alt_encoder_eval --encoders msmarco medcpt   # subset

Notes:
  - MedCPT is asymmetric: different encoder for query vs document.
  - multi-qa-mpnet is symmetric: same encoder for both.
  - msmarco-distilbert-base-v2 is symmetric: same encoder for both (our baseline).
  - Encoding 4194 docs per model takes ~3-5 min on CPU.
  - Embeddings are cached to disk to avoid re-encoding (results/phase1/tuning/embeddings/).
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv()

from src.data.loader import load_corpus
from src.evaluation.evaluator import build_query
from src.evaluation.metrics import (
    average_precision,
    mean_average_precision,
    mean_reciprocal_rank,
    precision_at_k,
    reciprocal_rank,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR  = ROOT / "results" / "phase1" / "tuning"
OUTPUT_CSV  = OUTPUT_DIR / "encoder_comparison.csv"
EMB_CACHE   = OUTPUT_DIR / "embeddings"   # cached encoder embeddings
QUERY_FIELD = "concatenated"


# ---------------------------------------------------------------------------
# Generic encoder utilities (same pattern as encoder.py)
# ---------------------------------------------------------------------------

def _mean_pool(model_output, attention_mask: torch.Tensor) -> torch.Tensor:
    token_emb = model_output.last_hidden_state
    mask_exp  = attention_mask.unsqueeze(-1).expand(token_emb.size()).float()
    return torch.sum(token_emb * mask_exp, 1) / torch.clamp(mask_exp.sum(1), min=1e-9)


def encode_texts(
    model_name: str,
    texts: list[str],
    batch_size: int = 32,
    device: str | None = None,
    prefix: str = "",
) -> np.ndarray:
    """
    Encode a list of texts with any HuggingFace AutoModel.
    Returns L2-normalised embeddings of shape (N, hidden_size).

    Args:
        prefix: Optional prefix to prepend to each text (used by some models like
                MedCPT which expects a specific format — pass "" for no prefix).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name).eval().to(device)

    all_embs = []
    for start in range(0, len(texts), batch_size):
        batch = [prefix + t for t in texts[start: start + batch_size]]
        enc   = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        enc   = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc, return_dict=True)

        emb = _mean_pool(out, enc["attention_mask"])
        emb = F.normalize(emb, p=2, dim=1)
        all_embs.append(emb.cpu().numpy())

    return np.vstack(all_embs)


# ---------------------------------------------------------------------------
# Encoder config registry
# ---------------------------------------------------------------------------

ENCODER_CONFIGS = {
    "msmarco": {
        "display":     "msmarco-distilbert-base-v2",
        "query_model": "sentence-transformers/msmarco-distilbert-base-v2",
        "doc_model":   "sentence-transformers/msmarco-distilbert-base-v2",
        "query_prefix": "",
        "doc_prefix":   "",
        "note":        "Current baseline (web search domain)",
    },
    "medcpt": {
        "display":     "MedCPT (asymmetric)",
        "query_model": "ncbi/MedCPT-Query-Encoder",
        "doc_model":   "ncbi/MedCPT-Article-Encoder",
        "query_prefix": "",
        "doc_prefix":   "",
        "note":        "Biomedical bi-encoder trained on PubMed click data",
    },
    "multi-qa": {
        "display":     "multi-qa-mpnet-base-cos-v1",
        "query_model": "sentence-transformers/multi-qa-mpnet-base-cos-v1",
        "doc_model":   "sentence-transformers/multi-qa-mpnet-base-cos-v1",
        "query_prefix": "",
        "doc_prefix":   "",
        "note":        "Trained on 215M QA pairs - strong general QA alignment",
    },
}


# ---------------------------------------------------------------------------
# Embedding cache helpers
# ---------------------------------------------------------------------------

def _cache_path(encoder_key: str, role: str) -> Path:
    """Cache path for pre-computed embeddings: tuning/embeddings/{encoder}_{role}.npy"""
    return EMB_CACHE / f"{encoder_key}_{role}.npy"


def load_or_encode_docs(
    encoder_key: str,
    corpus: list[dict],
    force: bool = False,
) -> np.ndarray:
    """Load corpus embeddings from cache or encode from scratch."""
    cfg   = ENCODER_CONFIGS[encoder_key]
    path  = _cache_path(encoder_key, "docs")

    if path.exists() and not force:
        embs = np.load(path)
        print(f"[{encoder_key}] Loaded doc embeddings from cache: {embs.shape}")
        return embs

    print(f"[{encoder_key}] Encoding {len(corpus)} docs with '{cfg['doc_model']}'...")
    t0 = time.time()
    texts = [doc["contents"] for doc in corpus]
    embs  = encode_texts(cfg["doc_model"], texts, prefix=cfg["doc_prefix"])
    elapsed = time.time() - t0
    print(f"[{encoder_key}] Encoded {len(corpus)} docs in {elapsed:.1f}s. Shape: {embs.shape}")

    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embs)
    print(f"[{encoder_key}] Saved to {path}")
    return embs


def load_or_encode_queries(
    encoder_key: str,
    topics: list[dict],
    query_field: str,
    force: bool = False,
) -> np.ndarray:
    """Load query embeddings from cache or encode from scratch."""
    cfg  = ENCODER_CONFIGS[encoder_key]
    role = f"queries_{query_field}"
    path = _cache_path(encoder_key, role)

    if path.exists() and not force:
        embs = np.load(path)
        print(f"[{encoder_key}] Loaded query embeddings from cache: {embs.shape}")
        return embs

    queries = [build_query(t, query_field) for t in topics]
    print(f"[{encoder_key}] Encoding {len(queries)} queries with '{cfg['query_model']}'...")
    embs = encode_texts(cfg["query_model"], queries, prefix=cfg["query_prefix"])
    print(f"[{encoder_key}] Query embeddings shape: {embs.shape}")

    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embs)
    return embs


# ---------------------------------------------------------------------------
# Pure-Python IR evaluation (no OpenSearch)
# ---------------------------------------------------------------------------

def evaluate_encoder_on_topics(
    doc_embs:     np.ndarray,         # (N_docs, dim) — L2-normalised
    query_embs:   np.ndarray,         # (N_queries, dim) — L2-normalised
    topics:       list[dict],
    qrels:        dict[str, dict],
    all_doc_ids:  list[str],
    top_k:        int = 100,
) -> dict:
    """
    Evaluate an encoder on a set of topics using exact cosine similarity.

    No OpenSearch — pure matrix multiplication.

    Returns:
        {"MAP": float, "MRR": float, "P@10": float, "per_query": {topic_id: {AP, RR, P@10}}}
    """
    # Brute-force cosine similarity: (N_queries, N_docs)
    # Both sides are L2-normalised -- dot product = cosine similarity
    scores_matrix = query_embs @ doc_embs.T     # shape: (N_queries, N_docs)

    all_queries_for_map = []
    per_query = {}

    for q_idx, topic in enumerate(topics):
        topic_id  = str(topic["id"])
        qrels_set = set(qrels.get(topic_id, {}).keys())

        # Get top-k document indices by cosine similarity
        scores    = scores_matrix[q_idx]        # shape: (N_docs,)
        top_k_idx = np.argpartition(scores, -top_k)[-top_k:]
        top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]   # sort descending

        # Build ranking (list of doc indices, ordered by rank)
        ranking   = top_k_idx.tolist()

        # Build relevance array: relevance[i] = True iff all_doc_ids[i] in qrels_set
        relevance = [doc_id in qrels_set for doc_id in all_doc_ids]

        all_queries_for_map.append((relevance, ranking))

        per_query[topic_id] = {
            "AP":   average_precision(ranking, relevance),
            "RR":   reciprocal_rank(ranking, relevance),
            "P@10": precision_at_k(ranking, relevance, 10),
        }

    return {
        "MAP":       mean_average_precision(all_queries_for_map),
        "MRR":       mean_reciprocal_rank(all_queries_for_map),
        "P@10":      float(np.mean([v["P@10"] for v in per_query.values()])) if per_query else 0.0,
        "per_query": per_query,
    }


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def run_encoder_comparison(
    corpus:       list[dict],
    train_topics: list[dict],
    qrels:        dict,
    all_doc_ids:  list[str],
    encoder_keys: list[str] | None = None,
    force_encode: bool = False,
) -> list[dict]:
    """
    Compare a set of encoders on the full train set (no CV — just MAP on all 32 train topics).

    For a more rigorous comparison, run evaluate_encoder_on_topics on each
    fold of make_folds(train_topics, 5) instead.

    Returns a list of result dicts sorted by MAP descending.
    """
    if encoder_keys is None:
        encoder_keys = list(ENCODER_CONFIGS.keys())

    results = []

    print(f"\n{'Encoder':<30} | {'MAP':>8} | {'MRR':>8} | {'P@10':>8} | Note")
    print("-" * 75)

    for key in encoder_keys:
        cfg = ENCODER_CONFIGS[key]
        print(f"\n[{key}] Loading '{cfg['display']}'...")

        doc_embs   = load_or_encode_docs(key, corpus, force=force_encode)
        query_embs = load_or_encode_queries(key, train_topics, QUERY_FIELD, force=force_encode)

        metrics = evaluate_encoder_on_topics(
            doc_embs, query_embs, train_topics, qrels, all_doc_ids
        )

        print(
            f"  {cfg['display']:<30} | {metrics['MAP']:>8.4f} | "
            f"{metrics['MRR']:>8.4f} | {metrics['P@10']:>8.4f} | {cfg['note']}"
        )

        results.append({
            "encoder":  key,
            "display":  cfg["display"],
            "map":      metrics["MAP"],
            "mrr":      metrics["MRR"],
            "p10":      metrics["P@10"],
            "note":     cfg["note"],
        })

    results_sorted = sorted(results, key=lambda r: r["map"], reverse=True)
    best = results_sorted[0]
    baseline = next((r for r in results if r["encoder"] == "msmarco"), results_sorted[-1])

    print(f"\n{'=' * 75}")
    print(f">> Best encoder: {best['display']}  (MAP={best['map']:.4f})")
    if best["encoder"] != "msmarco":
        print(f"  Baseline (msmarco): MAP={baseline['map']:.4f}")
        print(f"  Improvement on train: Δ MAP = {best['map'] - baseline['map']:+.4f}")
    print()
    print("Next step: if the best encoder beats msmarco by >0.02 MAP,")
    print("  add an embedding field to OpenSearch and re-evaluate with HNSW KNN.")
    print("  See FINE_TUNE.md §8.3 Step 4 for instructions.")

    return results_sorted


def save_results(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved: {path}")


def load_and_print_results(path: Path) -> None:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    rows_sorted = sorted(rows, key=lambda r: float(r["map"]), reverse=True)
    print(f"\nEncoder comparison results (loaded from {path.name}):")
    print(f"{'Encoder':<35} | {'MAP':>8} | {'MRR':>8} | {'P@10':>8}")
    print("-" * 64)
    for row in rows_sorted:
        marker = " <- best" if row == rows_sorted[0] else ""
        print(
            f"{row['display']:<35} | {float(row['map']):>8.4f} | "
            f"{float(row['mrr']):>8.4f} | {float(row['p10']):>8.4f}{marker}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Convenience entry-point callable from notebooks or other scripts
# ---------------------------------------------------------------------------

def run_encoder_comparison_and_save(
    corpus: list[dict] | None = None,
    train_topics: list[dict] | None = None,
    qrels: dict | None = None,
    all_doc_ids: list[str] | None = None,
    output_csv: str | Path = OUTPUT_CSV,
    encoder_keys: list[str] | None = None,
    force_encode: bool = False,
) -> list[dict]:
    """
    Compare dense encoders on the train set and save results to CSV. Always overwrites existing CSV.

    Args:
        corpus:        list of corpus dicts (loaded from disk if None)
        train_topics:  list of train topic dicts (loaded from splits if None)
        qrels:         binary qrels dict (loaded from disk if None)
        all_doc_ids:   list of all corpus PMIDs (derived from corpus if None)
        output_csv:    path to write CSV results
        encoder_keys:  which encoders to evaluate (default: all in ENCODER_CONFIGS)
        force_encode:  re-encode embeddings even if cached .npy files exist

    Returns:
        list of result dicts sorted by MAP descending
    """
    output_csv = Path(output_csv)

    if corpus is None:
        corpus = load_corpus(ROOT / "data" / "filtered_pubmed_abstracts.txt")
    if all_doc_ids is None:
        all_doc_ids = [doc["id"] for doc in corpus]
    if train_topics is None:
        with open(ROOT / "results" / "splits" / "train_queries.json") as f:
            train_topics = json.load(f)
    if qrels is None:
        with open(ROOT / "results" / "qrels.json") as f:
            qrels = json.load(f)

    print("=" * 75)
    print("Dense Encoder Comparison — exact cosine similarity on train set")
    print(f"Train topics: {len(train_topics)}  |  Corpus: {len(corpus)} docs")
    print("=" * 75)

    results = run_encoder_comparison(
        corpus, train_topics, qrels, all_doc_ids,
        encoder_keys=encoder_keys,
        force_encode=force_encode,
    )
    save_results(results, output_csv)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Compare dense encoders on train set")
    parser.add_argument("--show",     action="store_true", help="Load and print existing CSV")
    parser.add_argument("--reenc",    action="store_true", help="Force re-encode embeddings")
    parser.add_argument(
        "--encoders", nargs="+", default=None,
        choices=list(ENCODER_CONFIGS.keys()),
        help="Which encoders to compare (default: all)",
    )
    args = parser.parse_args()

    if args.show:
        if OUTPUT_CSV.exists():
            load_and_print_results(OUTPUT_CSV)
        else:
            print(f"No results file at {OUTPUT_CSV}. Run without --show first.")
        sys.exit(0)

    run_encoder_comparison_and_save(
        encoder_keys=args.encoders,
        force_encode=args.reenc,
    )
