
import logging
import math

import numpy as np

from src.data.query_builder import build_query
from src.evaluation.metrics import (
    average_precision,
    mean_average_precision,
    mean_ndcg_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
    results_to_ranking,
    results_to_ranking_graded,
)

logger = logging.getLogger(__name__)


# Fold construction
def make_folds(topics: list[dict], n_folds: int = 5) -> list[tuple[list[dict], list[dict]]]:
    """Split topics into n_folds (train_fold, val_fold) pairs, sorted by ID for reproducibility."""
    sorted_topics = sorted(topics, key=lambda t: t["id"])
    n = len(sorted_topics)
    fold_size = math.ceil(n / n_folds)
    folds = []
    for fold_idx in range(n_folds):
        start = fold_idx * fold_size
        end   = min(start + fold_size, n)
        val   = sorted_topics[start:end]
        train = sorted_topics[:start] + sorted_topics[end:]
        folds.append((train, val))
    return folds


# Single-fold evaluation
def evaluate_fold(
    retriever,
    val_topics:   list[dict],
    qrels:        dict[str, dict],
    qrels_graded: dict[str, dict],
    all_doc_ids:  list[str],
    query_field:  str = "concatenated",
    size:         int = 100,
) -> dict:
    """
    Evaluate a retriever on a held-out fold. Primary metric: NDCG@100 (graded qrels).

    Args:
        retriever:    retriever with .search(query, size) -> [(pmid, score), ...]
        val_topics:   held-out topics for this fold
        qrels:        {topic_id: {pmid: 1}} binary qrels (for MAP/MRR)
        qrels_graded: {topic_id: {pmid: 0/1/2}} graded qrels (for NDCG@100)
        all_doc_ids:  full corpus doc ID list
        query_field:  "topic", "question", or "concatenated"
        size:         number of docs to retrieve per query

    Returns:{"NDCG@100": float, "MAP": float, "MRR": float, "P@10": float, "per_query": {...}}
    """
    all_binary = []
    all_graded = []
    per_query  = {}

    for topic in val_topics:
        tid       = str(topic["id"])
        qrels_set = set(qrels.get(tid, {}).keys())

        results = retriever.search(build_query(topic, query_field), size=size)

        # binary path — MAP/MRR
        relevance, ranking = results_to_ranking(results, qrels_set, all_doc_ids)
        all_binary.append((relevance, ranking))

        # graded path — NDCG@100
        scores_g, ranking_g = results_to_ranking_graded(
            results, qrels_graded.get(tid, {}), all_doc_ids
        )
        all_graded.append((scores_g, ranking_g))

        per_query[tid] = {
            "NDCG@100": ndcg_at_k(ranking_g, scores_g, 100),
            "AP":      average_precision(ranking, relevance),
            "RR":      reciprocal_rank(ranking, relevance),
            "P@10":    precision_at_k(ranking, relevance, 10),
        }

    return {
        "NDCG@100": mean_ndcg_at_k(all_graded, k=100),
        "MAP":     mean_average_precision(all_binary),
        "MRR":     mean_reciprocal_rank(all_binary),
        "P@10":    float(np.mean([v["P@10"] for v in per_query.values()])) if per_query else 0.0,
        "per_query": per_query,
    }


# Full k-fold CV loop
def run_cv(
    retriever_factory,
    topics:       list[dict],
    qrels:        dict[str, dict],
    qrels_graded: dict[str, dict],
    all_doc_ids:  list[str],
    query_field:  str = "concatenated",
    n_folds:      int = 5,
    size:         int = 100,
    verbose:      bool = True,
) -> dict:
    """
    K-fold CV for a retriever. Primary criterion: NDCG@100. MAP also tracked.

    Returns:
        {
            "ndcg_per_fold": [float, ...],  # NDCG@100 for each fold (primary metric)
            "map_per_fold":  [float, ...],
            "mrr_per_fold":  [float, ...],
            "p10_per_fold":  [float, ...],
            "mean_ndcg":     float,         # sort key in sweepers
            "std_ndcg":      float,
            "mean_map":      float,
            "std_map":       float,
            "mean_mrr":      float,
            "mean_p10":      float,
        }
    """
    folds = make_folds(topics, n_folds)

    ndcg_per_fold = []
    map_per_fold  = []
    mrr_per_fold  = []
    p10_per_fold  = []

    for fold_idx, (_, val_fold) in enumerate(folds):
        result = evaluate_fold(
            retriever_factory(), val_fold, qrels, qrels_graded, all_doc_ids, query_field, size
        )
        ndcg_per_fold.append(result["NDCG@100"])
        map_per_fold.append(result["MAP"])
        mrr_per_fold.append(result["MRR"])
        p10_per_fold.append(result["P@10"])

        if verbose:
            logger.info(
                "Fold %d/%d  val=%d  NDCG@100=%.4f  MAP=%.4f  MRR=%.4f",
                fold_idx + 1, n_folds, len(val_fold),
                result["NDCG@100"], result["MAP"], result["MRR"],
            )

    return {
        "ndcg_per_fold": ndcg_per_fold,
        "map_per_fold":  map_per_fold,
        "mrr_per_fold":  mrr_per_fold,
        "p10_per_fold":  p10_per_fold,
        "mean_ndcg":     float(np.mean(ndcg_per_fold)),
        "std_ndcg":      float(np.std(ndcg_per_fold)),
        "mean_map":      float(np.mean(map_per_fold)),
        "std_map":       float(np.std(map_per_fold)),
        "mean_mrr":      float(np.mean(mrr_per_fold)),
        "mean_p10":      float(np.mean(p10_per_fold)),
    }


######################################################################
## 
##                      LOCAL TEST
## run: python -m src.tuning.cv_utils
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
    with open(ROOT / "results" / "qrels" / "qrels.json") as f:
        qrels = json.load(f)
    with open(ROOT / "results" / "qrels" / "qrels_graded.json") as f:
        qrels_graded = json.load(f)

    from src.data.loader import load_corpus
    corpus = load_corpus(ROOT / "data" / "filtered_pubmed_abstracts.txt")
    all_doc_ids = [doc["id"] for doc in corpus]

    from src.indexing.opensearch_client import get_client
    from src.retrieval.bm25 import BM25Retriever

    client = get_client()
    index_name = os.getenv("OPENSEARCH_INDEX", "")

    print("=" * 60)
    print("cv_utils self-test -- BM25 5-fold CV on train set")
    print("=" * 60)

    folds = make_folds(train_topics, n_folds=5)
    print(f"\nmake_folds(n=32, n_folds=5):")
    for i, (tr, val) in enumerate(folds):
        ids = [t["id"] for t in val]
        print(f"  Fold {i+1}: train={len(tr)}, val={len(val)}, val_ids={ids}")

    print("\nRunning BM25 5-fold CV (this takes ~30s)...")
    factory = lambda: BM25Retriever(client, index_name)
    cv = run_cv(factory, train_topics, qrels, qrels_graded, all_doc_ids, verbose=True)

    print(f"\nBM25 CV Results:")
    print(f"  NDCG@100 per fold : {[f'{v:.4f}' for v in cv['ndcg_per_fold']]}")
    print(f"  Mean NDCG@100     : {cv['mean_ndcg']:.4f} +/- {cv['std_ndcg']:.4f}")
    print(f"  MAP per fold     : {[f'{v:.4f}' for v in cv['map_per_fold']]}")
    print(f"  Mean MAP         : {cv['mean_map']:.4f} +/- {cv['std_map']:.4f}")
    print(f"  Mean MRR         : {cv['mean_mrr']:.4f}")
    print(f"  Mean P@10        : {cv['mean_p10']:.4f}")
    print("\n[ok] cv_utils self-test passed")
