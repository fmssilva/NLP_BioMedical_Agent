"""
src/tuning/cv_utils.py

K-fold cross-validation utilities for IR hyperparameter tuning.

The train set has only 32 topics — a single train/val split gives noisy MAP estimates
because one "hard" topic in the val set can swing MAP by ±0.02. K-fold CV averages
over multiple splits to give a more robust estimate.

For retrieval hyperparameters (BM25 k1/b, LM-Dir μ), we just split the TOPICS into
folds and measure MAP on the held-out fold with each hyperparameter setting. No
training occurs — just evaluation with different parameter values.

Public API:
    make_folds(topics, n_folds=5)           -> list of (train_fold, val_fold)
    evaluate_fold(retriever, val_topics, qrels, all_doc_ids, query_field)
                                            -> {"MAP": float, "MRR": float, "P@10": float}
    run_cv(retriever_factory, topics, qrels, all_doc_ids, query_field, n_folds=5)
                                            -> {"map_per_fold": [...], "mean_map": float, "std_map": float}
"""

import logging
import math

import numpy as np

from src.evaluation.evaluator import build_query
from src.evaluation.metrics import (
    average_precision,
    mean_average_precision,
    mean_reciprocal_rank,
    precision_at_k,
    reciprocal_rank,
    results_to_ranking,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fold construction
# ---------------------------------------------------------------------------

def make_folds(topics: list[dict], n_folds: int = 5) -> list[tuple[list[dict], list[dict]]]:
    """
    Split topics into n_folds (train_fold, val_fold) pairs.

    Topics are shuffled by their ID to ensure stable, reproducible splits.
    With 32 train topics and 5 folds: 4 folds of 6 topics + 1 fold of 8 topics.

    Args:
        topics:  list of topic dicts with "id" key.
        n_folds: number of folds (default 5).

    Returns:
        list of (train_topics, val_topics) tuples — one per fold.
    """
    # sort by ID for reproducibility (not random)
    sorted_topics = sorted(topics, key=lambda t: t["id"])
    n = len(sorted_topics)

    folds = []
    fold_size = math.ceil(n / n_folds)

    for fold_idx in range(n_folds):
        start = fold_idx * fold_size
        end   = min(start + fold_size, n)

        val_topics   = sorted_topics[start:end]
        train_topics = sorted_topics[:start] + sorted_topics[end:]

        folds.append((train_topics, val_topics))

    return folds


# ---------------------------------------------------------------------------
# Single-fold evaluation (no training — just measure MAP on val topics)
# ---------------------------------------------------------------------------

def evaluate_fold(
    retriever,
    val_topics: list[dict],
    qrels: dict[str, dict],
    all_doc_ids: list[str],
    query_field: str = "concatenated",
    size: int = 100,
) -> dict:
    """
    Evaluate a retriever on a set of topics and return aggregate metrics.

    For retrieval hyperparameter tuning, the "retriever" is already configured
    with the hyperparameter value being tested — this function just runs it.

    Args:
        retriever:    a retriever with .search(query, size) -> [(pmid, score), ...]
        val_topics:   topics to evaluate on (the held-out fold)
        qrels:        {topic_id: {pmid: 1}} relevance judgements
        all_doc_ids:  full list of corpus doc IDs (for results_to_ranking)
        query_field:  "topic", "question", or "concatenated"
        size:         number of results to retrieve per query

    Returns:
        {"MAP": float, "MRR": float, "P@10": float, "per_query": {topic_id: {AP, RR, P@10}}}
    """
    all_queries = []
    per_query = {}

    for topic in val_topics:
        topic_id = str(topic["id"])
        qrels_set = set(qrels.get(topic_id, {}).keys())

        query = build_query(topic, query_field)
        results = retriever.search(query, size=size)

        relevance, ranking = results_to_ranking(results, qrels_set, all_doc_ids)
        all_queries.append((relevance, ranking))

        per_query[topic_id] = {
            "AP":   average_precision(ranking, relevance),
            "RR":   reciprocal_rank(ranking, relevance),
            "P@10": precision_at_k(ranking, relevance, 10),
        }

    return {
        "MAP":       mean_average_precision(all_queries),
        "MRR":       mean_reciprocal_rank(all_queries),
        "P@10":      float(np.mean([v["P@10"] for v in per_query.values()])) if per_query else 0.0,
        "per_query": per_query,
    }


# ---------------------------------------------------------------------------
# Full k-fold CV loop
# ---------------------------------------------------------------------------

def run_cv(
    retriever_factory,
    topics: list[dict],
    qrels: dict[str, dict],
    all_doc_ids: list[str],
    query_field: str = "concatenated",
    n_folds: int = 5,
    size: int = 100,
    verbose: bool = True,
) -> dict:
    """
    Run k-fold cross-validation for a retriever configuration.

    retriever_factory is a CALLABLE that creates a fresh retriever instance.
    It is called once per fold. This allows stateless retrievers to be
    instantiated with different parameters per call.

    Args:
        retriever_factory: callable () -> retriever (e.g. lambda: BM25Retriever(client, index))
        topics:            train topics to split (usually 32 odd-ID topics)
        qrels:             relevance judgements
        all_doc_ids:       full corpus doc ID list
        query_field:       query formulation field
        n_folds:           number of folds (default 5)
        size:              results per query
        verbose:           print fold-by-fold MAP if True

    Returns:
        {
            "map_per_fold":  [float, ...],   # MAP for each held-out fold
            "mrr_per_fold":  [float, ...],
            "p10_per_fold":  [float, ...],
            "mean_map":      float,
            "std_map":       float,
            "mean_mrr":      float,
            "mean_p10":      float,
        }
    """
    folds = make_folds(topics, n_folds)

    map_per_fold  = []
    mrr_per_fold  = []
    p10_per_fold  = []

    for fold_idx, (train_fold, val_fold) in enumerate(folds):
        retriever = retriever_factory()
        result = evaluate_fold(retriever, val_fold, qrels, all_doc_ids, query_field, size)

        map_per_fold.append(result["MAP"])
        mrr_per_fold.append(result["MRR"])
        p10_per_fold.append(result["P@10"])

        if verbose:
            logger.info(
                "Fold %d/%d — val topics: %d — MAP=%.4f  MRR=%.4f  P@10=%.4f",
                fold_idx + 1, n_folds, len(val_fold),
                result["MAP"], result["MRR"], result["P@10"],
            )

    return {
        "map_per_fold":  map_per_fold,
        "mrr_per_fold":  mrr_per_fold,
        "p10_per_fold":  p10_per_fold,
        "mean_map":      float(np.mean(map_per_fold)),
        "std_map":       float(np.std(map_per_fold)),
        "mean_mrr":      float(np.mean(mrr_per_fold)),
        "mean_p10":      float(np.mean(p10_per_fold)),
    }


# ---------------------------------------------------------------------------
# Self-test: python -m src.tuning.cv_utils
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    import os
    from pathlib import Path

    from dotenv import load_dotenv

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load_dotenv()

    ROOT = Path(__file__).resolve().parents[2]

    # Load train topics and qrels
    with open(ROOT / "results" / "splits" / "train_queries.json") as f:
        train_topics = json.load(f)
    with open(ROOT / "results" / "qrels.json") as f:
        qrels = json.load(f)

    from src.data.loader import load_corpus
    corpus = load_corpus(ROOT / "data" / "filtered_pubmed_abstracts.txt")
    all_doc_ids = [doc["id"] for doc in corpus]

    from src.indexing.opensearch_client import get_client
    from src.retrieval.bm25 import BM25Retriever

    client = get_client()
    index_name = os.getenv("OPENSEARCH_INDEX", "")

    print("=" * 60)
    print("cv_utils self-test — BM25 5-fold CV on train set")
    print("=" * 60)

    folds = make_folds(train_topics, n_folds=5)
    print(f"\nmake_folds(n=32, n_folds=5):")
    for i, (tr, val) in enumerate(folds):
        ids = [t["id"] for t in val]
        print(f"  Fold {i+1}: train={len(tr)}, val={len(val)}, val_ids={ids}")

    print("\nRunning BM25 5-fold CV (this takes ~30s)...")
    factory = lambda: BM25Retriever(client, index_name)
    cv = run_cv(factory, train_topics, qrels, all_doc_ids, verbose=True)

    print(f"\nBM25 CV Results:")
    print(f"  MAP per fold: {[f'{v:.4f}' for v in cv['map_per_fold']]}")
    print(f"  Mean MAP:     {cv['mean_map']:.4f} ± {cv['std_map']:.4f}")
    print(f"  Mean MRR:     {cv['mean_mrr']:.4f}")
    print(f"  Mean P@10:    {cv['mean_p10']:.4f}")
    print("\n[ok] cv_utils self-test passed")
