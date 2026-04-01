"""
src/evaluation/evaluator.py

Evaluation pipeline for Phase 1 — runs retrieval and computes metrics.

Responsibilities:
  1. Query field ablation on train set (topic / question / concatenated)
  2. LM-JM lambda selection on train set (lambda=0.1 vs lambda=0.7)
  3. Full evaluation of all 5 strategies on train or test set
  4. Save run files to results/phase1/

Public API:
    build_query(topic, field)                        -- format a query string from a topic
    run_strategy(retriever, queries, qrels, all_doc_ids, query_field)
                                                     -- search + metrics for one strategy
    field_ablation(client, index, corpus_ids, train_topics, qrels)
                                                     -- BM25 on 3 query fields, pick best
    evaluate_all_strategies(client, index, encoder, corpus_ids, topics, qrels, query_field, lmjm_variant)
                                                     -- full 5-strategy evaluation table
    save_run(run, path)                              -- save {topic_id: [(pmid, score)...]} to JSON
    load_run(path)                                   -- load run file, restore tuples

This is a plain Python pipeline file, not a notebook. Run with:
    python -m src.evaluation.evaluator
"""

import json
import logging
import os
from pathlib import Path

import numpy as np

from src.data.loader import load_corpus, load_topics
from src.data.qrels_builder import build_qrels
from src.embeddings.corpus_encoder import load_embeddings
from src.embeddings.encoder import Encoder
from src.evaluation.metrics import (
    average_precision,
    mean_average_precision,
    mean_pr_curve,
    mean_reciprocal_rank,
    precision_at_k,
    reciprocal_rank,
    results_to_ranking,
)
from src.indexing.opensearch_client import get_client
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.knn import KNNRetriever
from src.retrieval.lm_dirichlet import LMDirichletRetriever
from src.retrieval.lm_jelinek_mercer import LMJMRetriever
from src.retrieval.rrf import RRFRetriever

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Query formatting
# ---------------------------------------------------------------------------

# Format a topic dict into a query string for the chosen field.
def build_query(topic: dict, field: str) -> str:
    """
    Return a query string from a topic dict.

    Args:
        topic: dict with keys 'topic', 'question', 'narrative'.
        field: one of 'topic', 'question', 'concatenated'.
    """
    if field == "topic":
        return topic["topic"]
    if field == "question":
        return topic["question"]
    if field == "concatenated":
        return f"{topic['topic']} {topic['question']} {topic['narrative']}"
    raise ValueError(f"Unknown query field: '{field}'. Use 'topic', 'question', or 'concatenated'.")


# ---------------------------------------------------------------------------
# Single-strategy run (search all queries, compute metrics)
# ---------------------------------------------------------------------------

def run_strategy(
    retriever,
    topics: list[dict],
    qrels: dict[str, dict],
    all_doc_ids: list[str],
    query_field: str,
    size: int = 100,
) -> dict:
    """
    Run a retriever on all topics and compute IR metrics.

    Returns a dict with keys:
        run         -- {topic_id: [(pmid, score), ...]}
        per_query   -- {topic_id: {"AP": float, "RR": float, "P@10": float}}
        MAP         -- float
        MRR         -- float
        P@10        -- float
        pr_curves   -- (recall_levels, mean_precisions) for plotting
    """
    run = {}
    per_query = {}
    all_queries_for_map = []  # list of (relevance_labels, ranking) for MAP/MRR/PR

    for topic in topics:
        topic_id = str(topic["id"])
        qrels_set = set(qrels.get(topic_id, {}).keys())

        query = build_query(topic, query_field)
        results = retriever.search(query, size=size)

        run[topic_id] = results

        relevance, ranking = results_to_ranking(results, qrels_set, all_doc_ids)
        all_queries_for_map.append((relevance, ranking))

        ap  = average_precision(ranking, relevance)
        rr  = reciprocal_rank(ranking, relevance)
        p10 = precision_at_k(ranking, relevance, 10)
        per_query[topic_id] = {"AP": ap, "RR": rr, "P@10": p10}

    # aggregate metrics
    map_score = mean_average_precision(all_queries_for_map)
    mrr_score = mean_reciprocal_rank(all_queries_for_map)
    p10_mean  = float(np.mean([v["P@10"] for v in per_query.values()]))
    rl, mp    = mean_pr_curve(all_queries_for_map)

    return {
        "run":       run,
        "per_query": per_query,
        "MAP":       map_score,
        "MRR":       mrr_score,
        "P@10":      p10_mean,
        "pr_curves": (rl, mp),
    }


# ---------------------------------------------------------------------------
# Query field ablation (BM25 only, train set)
# ---------------------------------------------------------------------------

def field_ablation(
    client,
    index_name: str,
    all_doc_ids: list[str],
    train_topics: list[dict],
    qrels: dict[str, dict],
) -> str:
    """
    Run BM25 with 3 query fields on the train set, pick the one with highest MAP.

    Prints a comparison table and returns the winning field name.
    """
    retriever = BM25Retriever(client, index_name)
    fields = ["topic", "question", "concatenated"]
    results = {}

    print("\n" + "=" * 50)
    print("Query Field Ablation (BM25, train set)")
    print("=" * 50)
    print(f"{'Field':>15} | {'MAP':>8} | {'MRR':>8} | {'P@10':>8}")
    print("-" * 50)

    for field in fields:
        r = run_strategy(retriever, train_topics, qrels, all_doc_ids, field)
        results[field] = r
        print(f"{field:>15} | {r['MAP']:>8.4f} | {r['MRR']:>8.4f} | {r['P@10']:>8.4f}")

    print("-" * 50)

    # pick winner by MAP
    winner = max(results, key=lambda f: results[f]["MAP"])
    print(f"\n  --> Best field: '{winner}'  (MAP={results[winner]['MAP']:.4f})")
    print(f"  --> Locking '{winner}' for all subsequent evaluations.")

    return winner


# ---------------------------------------------------------------------------
# LM-JM lambda selection (train set)
# ---------------------------------------------------------------------------

def lmjm_lambda_selection(
    client,
    index_name: str,
    all_doc_ids: list[str],
    train_topics: list[dict],
    qrels: dict[str, dict],
    query_field: str,
) -> str:
    """
    Compare LM-JM lambda=0.1 vs lambda=0.7 on train set.
    Returns the winning variant string: '01' or '07'.
    """
    print("\n" + "=" * 50)
    print("LM-JM Lambda Selection (train set)")
    print("=" * 50)
    print(f"{'Variant':>10} | {'lambda':>8} | {'MAP':>8} | {'MRR':>8} | {'P@10':>8}")
    print("-" * 50)

    variant_map = {"01": 0.1, "07": 0.7}
    results = {}

    for variant, lam in variant_map.items():
        retriever = LMJMRetriever(client, index_name, lambda_variant=variant)
        r = run_strategy(retriever, train_topics, qrels, all_doc_ids, query_field)
        results[variant] = r
        print(f"{'lmjm_' + variant:>10} | {lam:>8.1f} | {r['MAP']:>8.4f} | {r['MRR']:>8.4f} | {r['P@10']:>8.4f}")

    print("-" * 50)

    winner = max(results, key=lambda v: results[v]["MAP"])
    print(f"\n  --> Best LM-JM variant: lambda={variant_map[winner]:.1f} ('{winner}')")
    print(f"  --> Locking lambda={variant_map[winner]:.1f} for test evaluation.")

    return winner


# ---------------------------------------------------------------------------
# Full 5-strategy evaluation
# ---------------------------------------------------------------------------

def evaluate_all_strategies(
    client,
    index_name: str,
    encoder: Encoder,
    all_doc_ids: list[str],
    topics: list[dict],
    qrels: dict[str, dict],
    query_field: str,
    lmjm_variant: str,
    set_name: str = "train",
) -> dict[str, dict]:
    """
    Run all 5 strategies on the given topics and print a comparison table.

    Returns: {strategy_name: run_strategy_result_dict}
    """
    strategies = {
        "BM25":   BM25Retriever(client, index_name),
        "LM-JM":  LMJMRetriever(client, index_name, lambda_variant=lmjm_variant),
        "LM-Dir": LMDirichletRetriever(client, index_name),
        "KNN":    KNNRetriever(client, index_name, encoder=encoder),
        "RRF":    RRFRetriever(client, index_name, encoder=encoder),
    }

    print("\n" + "=" * 62)
    print(f"All-Strategy Evaluation ({set_name} set, field='{query_field}', lmjm={lmjm_variant})")
    print("=" * 62)
    print(f"{'Strategy':>10} | {'MAP':>8} | {'MRR':>8} | {'P@10':>8}")
    print("-" * 40)

    results = {}
    for name, retriever in strategies.items():
        print(f"  Running {name} ...", end="", flush=True)
        r = run_strategy(retriever, topics, qrels, all_doc_ids, query_field)
        results[name] = r
        print(f"\r{name:>10} | {r['MAP']:>8.4f} | {r['MRR']:>8.4f} | {r['P@10']:>8.4f}")

    print("-" * 40)

    # best strategy highlight
    best = max(results, key=lambda n: results[n]["MAP"])
    print(f"\n  --> Best strategy: {best}  (MAP={results[best]['MAP']:.4f})")

    return results


# ---------------------------------------------------------------------------
# Run file I/O
# ---------------------------------------------------------------------------

# Save a run dict {topic_id: [(pmid, score)...]} to JSON.
def save_run(run: dict, path: str | Path) -> None:
    """
    JSON serialises a run file. Tuples become lists (JSON limitation) — restoring
    on load is optional (evaluation code uses index [0] and [1]).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(run, f)
    print(f"[evaluator] Saved run file ({len(run)} topics) -> {path}")


# Load a run file and restore as list of [pmid, score] pairs.
def load_run(path: str | Path) -> dict:
    """Load a run file. Returns {topic_id: [[pmid, score], ...]}."""
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Convenience entry-point callable from notebooks or other scripts
# ---------------------------------------------------------------------------

def run_baseline_evaluation(
    client=None,
    index_name: str = "",
    encoder: "Encoder | None" = None,
    corpus: list[dict] | None = None,
    train_topics: list[dict] | None = None,
    test_topics: list[dict] | None = None,
    qrels: dict | None = None,
    output_dir: str | Path = Path(__file__).resolve().parents[2] / "results" / "phase1",
) -> dict:
    """
    Run the full baseline evaluation pipeline (field ablation, LM-JM selection,
    all-strategy evaluation on train + test) and save run files to output_dir.

    Always overwrites existing run files.

    Args:
        client:       OpenSearch client (created from env if None)
        index_name:   index to query (read from OPENSEARCH_INDEX env if empty)
        encoder:      Encoder instance for KNN/RRF (created with defaults if None)
        corpus:       list of corpus dicts (loaded from disk if None)
        train_topics: train topic list (loaded from disk if None)
        test_topics:  test topic list (loaded from disk if None)
        qrels:        binary qrels dict (loaded from disk if None)
        output_dir:   directory to write *_run.json and *_train_run.json files

    Returns:
        dict with keys 'train_results' and 'test_results', each mapping
        strategy name -> run_strategy result dict
    """
    import logging
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(name)s  %(message)s")

    output_dir = Path(output_dir)
    root = Path(__file__).resolve().parents[2]

    from dotenv import load_dotenv
    load_dotenv()

    if corpus is None:
        corpus = load_corpus(root / "data" / "filtered_pubmed_abstracts.txt")
    all_doc_ids = [doc["id"] for doc in corpus]
    print(f"Corpus: {len(corpus)} docs")

    if train_topics is None:
        with open(root / "results" / "splits" / "train_queries.json") as f:
            train_topics = json.load(f)
    if test_topics is None:
        with open(root / "results" / "splits" / "test_queries.json") as f:
            test_topics = json.load(f)
    print(f"Train queries: {len(train_topics)}, Test queries: {len(test_topics)}")

    if qrels is None:
        with open(root / "results" / "qrels.json") as f:
            qrels = json.load(f)
    print(f"Qrels topics: {len(qrels)}")

    if client is None:
        client = get_client()
    if not index_name:
        index_name = os.getenv("OPENSEARCH_INDEX", "")
    assert index_name, "OPENSEARCH_INDEX not set"

    if encoder is None:
        encoder = Encoder()

    print("=" * 62)
    print("Baseline Evaluation Pipeline (train set)")
    print("=" * 62)

    best_field = field_ablation(client, index_name, all_doc_ids, train_topics, qrels)
    best_lmjm  = lmjm_lambda_selection(client, index_name, all_doc_ids, train_topics, qrels, best_field)

    train_results = evaluate_all_strategies(
        client, index_name, encoder, all_doc_ids,
        train_topics, qrels, best_field, best_lmjm, set_name="train",
    )
    for name, result in train_results.items():
        safe_name = name.lower().replace("-", "_")
        save_run(result["run"], output_dir / f"{safe_name}_train_run.json")

    test_results = evaluate_all_strategies(
        client, index_name, encoder, all_doc_ids,
        test_topics, qrels, best_field, best_lmjm, set_name="test",
    )
    for name, result in test_results.items():
        safe_name = name.lower().replace("-", "_")
        save_run(result["run"], output_dir / f"{safe_name}_run.json")

    print("\n" + "=" * 62)
    print("Baseline evaluation complete.")
    print(f"  Best query field : {best_field}")
    print(f"  Best LM-JM lambda: {best_lmjm}")
    print(f"  Run files saved to: {output_dir}")
    print("=" * 62)

    return {"train_results": train_results, "test_results": test_results}


# ---------------------------------------------------------------------------
# Self-test / main pipeline: python -m src.evaluation.evaluator
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_baseline_evaluation()
