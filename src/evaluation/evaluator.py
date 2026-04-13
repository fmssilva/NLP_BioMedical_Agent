
import json
import logging
from pathlib import Path

import numpy as np

from src.data.query_builder import build_query  
from src.evaluation.metrics import (
    average_precision,
    mean_average_precision,
    mean_ndcg_at_k,
    mean_pr_curve,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    pr_curve,
    recall_at_k,
    reciprocal_rank,
    results_to_ranking,
    results_to_ranking_graded,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Live-retriever evaluation — all 5 metrics + PR curves
# ---------------------------------------------------------------------------

def evaluate_retriever(
    retriever,
    topics:       list[dict],
    qrels:        dict[str, dict], # binary
    qrels_graded: dict[str, dict], # graded 
    all_doc_ids:  list[str],
    query_field:  str = "topic+question",
    size:         int = 100,
) -> dict:
    """
    Run a retriever on every topic and compute all 5 metrics.

    Primary metric: NDCG@100 (graded qrels: supporting=2, neutral=1).
    Secondary: MAP, MRR, P@10, R@100.

    Returns:
        {
            "run":       {topic_id: [(pmid, score), ...]},
            "MAP":       float,
            "MRR":       float,
            "P@10":      float,
            "R@100":     float,
            "NDCG@100":  float,
            "pr_curves": (recall_levels, mean_precisions),
            "per_query": {topic_id: {"AP", "RR", "P@10", "R@100", "NDCG@100", "pr_curve"}},
        }
    """
    run        = {}
    per_query  = {}
    all_binary = []    # (relevance, ranking) for MAP/MRR/P@10/R@100
    all_graded = []    # (scores_graded, ranking_graded) for NDCG

    for topic in topics:
        tid       = str(topic["id"])
        qrels_set = set(qrels.get(tid, {}).keys())

        query   = build_query(topic, query_field)
        results = retriever.search(query, size=size)
        run[tid] = results

        relevance, ranking = results_to_ranking(results, qrels_set, all_doc_ids)
        all_binary.append((relevance, ranking))

        scores_g, ranking_g = results_to_ranking_graded(
            results, qrels_graded.get(tid, {}), all_doc_ids
        )
        all_graded.append((scores_g, ranking_g))

        q_recalls, q_precs = pr_curve(ranking, relevance)

        per_query[tid] = {
            "AP":       average_precision(ranking, relevance),
            "RR":       reciprocal_rank(ranking, relevance),
            "P@10":     precision_at_k(ranking, relevance, 10),
            "R@100":    recall_at_k(ranking, relevance, 100),
            "NDCG@100": ndcg_at_k(ranking_g, scores_g, 100),
            "pr_curve": (q_recalls, q_precs),
        }

    rl, mp = mean_pr_curve(all_binary)

    return {
        "run":       run,
        "MAP":       mean_average_precision(all_binary),
        "MRR":       mean_reciprocal_rank(all_binary),
        "P@10":      float(np.mean([v["P@10"]   for v in per_query.values()])),
        "R@100":     float(np.mean([v["R@100"]  for v in per_query.values()])),
        "NDCG@100":  mean_ndcg_at_k(all_graded, k=100),
        "pr_curves": (rl, mp),
        "per_query": per_query,
    }







########################## LOCAL TESTS   ####################################

def metrics_from_run(
    run:          dict,
    topics:       list[dict],
    qrels:        dict[str, dict], # binary
    qrels_graded: dict[str, dict], # graded
    all_doc_ids:  list[str],
) -> dict:
    """
    Recompute all 5 metrics from a pre-saved run dict.
    Returns:
        {"MAP", "MRR", "P@10", "R@100", "NDCG@100", "pr_curves", "per_query"}
    """
    per_query  = {}
    all_binary = []
    all_graded = []

    for topic in topics:
        tid       = str(topic["id"])
        qrels_set = set(qrels.get(tid, {}).keys())
        results   = [(pmid, float(score)) for pmid, score in run.get(tid, [])]

        relevance, ranking = results_to_ranking(results, qrels_set, all_doc_ids)
        all_binary.append((relevance, ranking))

        scores_g, ranking_g = results_to_ranking_graded(
            results, qrels_graded.get(tid, {}), all_doc_ids
        )
        all_graded.append((scores_g, ranking_g))

        q_recalls, q_precs = pr_curve(ranking, relevance)

        per_query[tid] = {
            "AP":       average_precision(ranking, relevance),
            "RR":       reciprocal_rank(ranking, relevance),
            "P@10":     precision_at_k(ranking, relevance, 10),
            "R@100":    recall_at_k(ranking, relevance, 100),
            "NDCG@100": ndcg_at_k(ranking_g, scores_g, 100),
            "pr_curve": (q_recalls, q_precs),
        }

    rl, mp = mean_pr_curve(all_binary)
    return {
        "MAP":       mean_average_precision(all_binary),
        "MRR":       mean_reciprocal_rank(all_binary),
        "P@10":      float(np.mean([v["P@10"]    for v in per_query.values()])),
        "R@100":     float(np.mean([v["R@100"]   for v in per_query.values()])),
        "NDCG@100":  mean_ndcg_at_k(all_graded, k=100),
        "pr_curves": (rl, mp),
        "per_query": per_query,
    }


# ---------------------------------------------------------------------------
# Run file I/O
# ---------------------------------------------------------------------------

def save_run(run: dict, path: str | Path) -> None:
    """
    Serialise a run dict {topic_id: [(pmid, score), ...]} to JSON.
    Tuples become lists (JSON limitation) — load_run restores them as lists.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(run, f)
    logger.info("[evaluator] Saved run file (%d topics) -> %s", len(run), path)


def load_run(path: str | Path) -> dict:
    """Load a run file. Returns {topic_id: [[pmid, score], ...]}."""
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Self-test: python -m src.evaluation.evaluator
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("evaluator.py — offline self-test")
    print("=" * 60)

    # ── Test build_query re-export ───────────────────────────────────────────
    _t = {
        "id": "1",
        "topic": "sleep apnea",
        "question": "What is the best treatment?",
        "narrative": "Documents about CPAP and surgery.",
    }
    assert build_query(_t, "topic") == "sleep apnea"
    assert build_query(_t, "topic+question") == "sleep apnea What is the best treatment?"
    assert build_query(_t, "concatenated") == (
        "sleep apnea What is the best treatment? Documents about CPAP and surgery."
    )
    print("  [ok] build_query re-export")

    # ── Test metrics_from_run with mock data ─────────────────────────────────
    _topics  = [{"id": str(i), "topic": f"t{i}", "question": "", "narrative": ""}
                for i in range(5)]
    _doc_ids = [f"doc{i:03d}" for i in range(20)]
    _qrels   = {str(i): {"doc000": 1, "doc001": 1, "doc002": 1} for i in range(5)}
    _qgr     = {str(i): {"doc000": 2, "doc001": 2, "doc002": 1} for i in range(5)}
    _run     = {str(i): [(_doc_ids[j], 1.0 - j * 0.05) for j in range(20)] for i in range(5)}

    m = metrics_from_run(_run, _topics, _qrels, _qgr, _doc_ids)
    assert "MAP" in m and "NDCG@100" in m and "R@100" in m
    assert 0.0 < m["NDCG@100"] <= 1.0
    assert 0.0 < m["MAP"] <= 1.0
    assert len(m["per_query"]) == 5
    print(f"  [ok] metrics_from_run  NDCG@100={m['NDCG@100']:.4f}  MAP={m['MAP']:.4f}")

    # ── Test save_run / load_run round-trip ─────────────────────────────────
    with tempfile.TemporaryDirectory() as _tmpdir:
        _path = Path(_tmpdir) / "test_run.json"
        save_run(_run, _path)
        assert _path.exists()
        _loaded = load_run(_path)
        assert set(_loaded.keys()) == set(_run.keys())
    print("  [ok] save_run / load_run round-trip")

    print("\n[ok] All evaluator self-tests passed.")

