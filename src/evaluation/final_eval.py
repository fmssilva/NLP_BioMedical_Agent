"""
src/evaluation/final_eval.py

Final Phase 1 evaluation with TUNED parameters on the test set.

After hyperparameter tuning on train (5-fold CV), we lock the best config
and evaluate once on the 33 test queries. This script does exactly that.

Relationship to evaluator.py:
  - evaluator.py handles baseline evaluation (train+test) with only binary
    metrics (MAP, MRR, P@10) and default index fields. It is used during
    the tuning phase for field ablation and lambda selection.
  - final_eval.py extends this with: (a) tuned index fields (BM25 k1=1.5,
    LM-Dir mu=75, MedCPT KNN), (b) full metric suite (adds R@100, NDCG@10),
    (c) graded qrels support, and (d) per-query PR curves.
  The two files share build_query() and save_run()/load_run() from evaluator.py
  but the evaluate_retriever() function here is a superset with richer output.

Tuned configuration (locked from train experiments) — see PHASE_1_BEST_CONFIG below.
Baseline configuration (for comparison) uses the default index fields: contents,
contents_lmdir (mu=2000), embedding (msmarco-distilbert).

Usage:
    python -m src.evaluation.final_eval              # run full eval
    python -m src.evaluation.final_eval --add-medcpt # also add MedCPT embeddings to index
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from src.data.loader import load_corpus, load_topics
from src.evaluation.evaluator import build_query, save_run, load_run
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
from src.indexing.index_builder import _BEST_PARAMS
from src.indexing.opensearch_client import get_client
from src.retrieval.base import FieldRetriever
from src.retrieval.rrf import rrf_merge

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Locked best configuration — set once here, used everywhere below
# These come from 5-fold CV on 32 train queries (see results/phase1/tuning/)
# ---------------------------------------------------------------------------

_lmjm_best  = _BEST_PARAMS["lmjm_lambdas"][0]          # 0.7
_lmdir_best = _BEST_PARAMS["lmdir_mus"][0]              # 75
_bm25_k1    = _BEST_PARAMS["bm25_k1_b_pairs"][0][0]    # 1.5
_bm25_b     = _BEST_PARAMS["bm25_k1_b_pairs"][0][1]    # 1.0

PHASE_1_BEST_CONFIG = {
    "query_field":  "concatenated",                # topic + question + narrative
    "lmjm_variant": str(_lmjm_best).replace(".", ""),          # "07"
    "lmjm_field":   f"contents_lmjm_{str(_lmjm_best).replace('.', '')}",  # contents_lmjm_07
    "lmdir_field":  f"contents_lmdir_{_lmdir_best}",           # contents_lmdir_75
    "bm25_field":   f"contents_bm25_k{str(_bm25_k1).replace('.','')}_b{str(_bm25_b).replace('.', '')}",  # contents_bm25_k15_b10
    "knn_field":    "embedding_medcpt",
    "encoder":      _BEST_PARAMS["encoders"][0][1],   # ncbi/MedCPT-Query-Encoder
}

# convenience aliases used throughout this file
QUERY_FIELD   = PHASE_1_BEST_CONFIG["query_field"]
LMJM_VARIANT  = PHASE_1_BEST_CONFIG["lmjm_variant"]
BEST_MU       = _lmdir_best
BEST_K1       = _bm25_k1
BEST_B        = _bm25_b
MEDCPT_EMB_PATH = ROOT / "results" / "phase1" / "tuning" / "embeddings" / "medcpt_docs.npy"
PHASE1_DIR    = ROOT / "results" / "phase1"


# ---------------------------------------------------------------------------
# MedCPT KNN retriever using OpenSearch embedding_medcpt field
# ---------------------------------------------------------------------------

class MedCPTKNNRetriever:
    """Dense KNN retrieval using MedCPT embeddings stored in the index."""

    def __init__(self, client, index_name: str, query_encoder=None):
        self.client = client
        self.index_name = index_name
        # lazy load MedCPT query encoder only when needed
        self._query_encoder = query_encoder
        self._tokenizer = None
        self._model = None

    def _ensure_encoder(self):
        """Load MedCPT query encoder on first use."""
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer
        import torch.nn.functional as F

        model_name = "ncbi/MedCPT-Query-Encoder"
        print(f"[medcpt_knn] Loading '{model_name}' ...")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).eval()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)

    def encode_query(self, text: str) -> np.ndarray:
        """Encode a single query with MedCPT-Query-Encoder, returns (768,) L2-normed."""
        import torch
        import torch.nn.functional as F

        self._ensure_encoder()
        enc = self._tokenizer(
            [text], padding=True, truncation=True,
            max_length=512, return_tensors="pt",
        )
        enc = {k: v.to(self._device) for k, v in enc.items()}
        with torch.no_grad():
            out = self._model(**enc, return_dict=True)
        # mean pooling + L2 normalize (same pattern as encoder.py)
        token_emb = out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).expand(token_emb.size()).float()
        pooled = torch.sum(token_emb * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        normed = F.normalize(pooled, p=2, dim=1)
        return normed[0].cpu().numpy()

    def search(self, query: str, size: int = 100) -> list[tuple[str, float]]:
        qvec = self.encode_query(query)
        body = {
            "size": size,
            "_source": ["doc_id"],
            "query": {
                "knn": {
                    "embedding_medcpt": {
                        "vector": qvec.tolist(),
                        "k": size,
                    }
                }
            },
        }
        resp = self.client.search(body=body, index=self.index_name)
        return [(h["_source"]["doc_id"], h["_score"]) for h in resp["hits"]["hits"]]


# ---------------------------------------------------------------------------
# Add MedCPT embedding field to the index
# ---------------------------------------------------------------------------

def add_medcpt_field(client, index_name: str) -> None:
    """Add a second knn_vector field for MedCPT embeddings (768-dim, same HNSW config)."""
    mapping = client.indices.get_mapping(index=index_name)
    actual_key = list(mapping.keys())[0]
    existing = set(mapping[actual_key]["mappings"]["properties"].keys())

    if "embedding_medcpt" in existing:
        print("[medcpt] Field 'embedding_medcpt' already exists — skipping.")
        return

    print("[medcpt] Adding 'embedding_medcpt' field to index...")
    # close index to update knn settings if needed
    new_field = {
        "embedding_medcpt": {
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
    }
    client.indices.put_mapping(index=index_name, body={"properties": new_field})
    print("[medcpt] Field added successfully.")


def populate_medcpt_embeddings(client, index_name: str, corpus: list[dict]) -> None:
    """Bulk-update all docs with MedCPT document embeddings from cached .npy file."""
    # check if already populated
    sample = client.search(
        index=index_name,
        body={"size": 1, "_source": ["embedding_medcpt"]},
    )
    hits = sample["hits"]["hits"]
    if hits and hits[0]["_source"].get("embedding_medcpt"):
        print("[medcpt] Embeddings already populated — skipping.")
        return

    if not MEDCPT_EMB_PATH.exists():
        raise FileNotFoundError(
            f"MedCPT doc embeddings not found at {MEDCPT_EMB_PATH}. "
            "Run: python -m src.tuning.alt_encoder_eval --encoders medcpt"
        )

    embs = np.load(MEDCPT_EMB_PATH)
    assert embs.shape[0] == len(corpus), f"Embedding count {embs.shape[0]} != corpus {len(corpus)}"
    print(f"[medcpt] Populating {len(corpus)} docs with MedCPT embeddings ({embs.shape})...")

    from opensearchpy.helpers import bulk

    def _actions():
        for i, doc in enumerate(corpus):
            yield {
                "_op_type": "update",
                "_index": index_name,
                "_id": doc["id"],
                "doc": {"embedding_medcpt": embs[i].tolist()},
            }

    t0 = time.time()
    success, errors = bulk(client, _actions(), chunk_size=100, raise_on_error=False)
    client.indices.refresh(index=index_name)
    elapsed = time.time() - t0
    print(f"[medcpt] Indexed {success} docs in {elapsed:.1f}s. Errors: {len(errors)}")


# ---------------------------------------------------------------------------
# Run a retriever on all topics, return full metrics
# ---------------------------------------------------------------------------

def evaluate_retriever(
    retriever,
    topics:       list[dict],
    qrels:        dict,
    qrels_graded: dict,
    all_doc_ids:  list[str],
    query_field:  str = QUERY_FIELD,
    size:         int = 100,
) -> dict:
    """
    Run a retriever on all topics and compute ALL 5 metrics + per-query details.

    Returns dict with: run, MAP, MRR, P@10, R@100, NDCG@10, pr_curves, per_query
    """
    run = {}
    per_query = {}
    all_binary  = []    # (relevance, ranking) for MAP/MRR/P@10/R@100
    all_graded  = []    # (scores_graded, ranking_graded) for NDCG

    for topic in topics:
        tid = str(topic["id"])
        qrels_set = set(qrels.get(tid, {}).keys())

        query = build_query(topic, query_field)
        results = retriever.search(query, size=size)
        run[tid] = results

        # binary metrics
        relevance, ranking = results_to_ranking(results, qrels_set, all_doc_ids)
        all_binary.append((relevance, ranking))

        # graded metrics
        scores_g, ranking_g = results_to_ranking_graded(
            results, qrels_graded.get(tid, {}), all_doc_ids
        )
        all_graded.append((scores_g, ranking_g))

        # per-topic PR curve (binary)
        q_recalls, q_precs = pr_curve(ranking, relevance)

        per_query[tid] = {
            "AP":      average_precision(ranking, relevance),
            "RR":      reciprocal_rank(ranking, relevance),
            "P@10":    precision_at_k(ranking, relevance, 10),
            "R@100":   recall_at_k(ranking, relevance, 100),
            "NDCG@10": ndcg_at_k(ranking_g, scores_g, 10),
            "pr_curve": (q_recalls, q_precs),
        }

    rl, mp = mean_pr_curve(all_binary)

    return {
        "run":       run,
        "MAP":       mean_average_precision(all_binary),
        "MRR":       mean_reciprocal_rank(all_binary),
        "P@10":      float(np.mean([v["P@10"]    for v in per_query.values()])),
        "R@100":     float(np.mean([v["R@100"]   for v in per_query.values()])),
        "NDCG@10":   mean_ndcg_at_k(all_graded, k=10),
        "pr_curves": (rl, mp),
        "per_query": per_query,
    }


# ---------------------------------------------------------------------------
# RRF from two retrievers
# ---------------------------------------------------------------------------

def evaluate_rrf(
    retriever_a,
    retriever_b,
    topics:       list[dict],
    qrels:        dict,
    qrels_graded: dict,
    all_doc_ids:  list[str],
    query_field:  str = QUERY_FIELD,
    rrf_k:        int = 60,
    size:         int = 100,
) -> dict:
    """Run two retrievers, merge with RRF, evaluate the fused ranking."""
    run = {}
    per_query = {}
    all_binary = []
    all_graded = []

    for topic in topics:
        tid = str(topic["id"])
        qrels_set = set(qrels.get(tid, {}).keys())
        query = build_query(topic, query_field)

        results_a = retriever_a.search(query, size=size)
        results_b = retriever_b.search(query, size=size)
        merged = rrf_merge(results_a, results_b, k=rrf_k)[:size]
        run[tid] = merged

        relevance, ranking = results_to_ranking(merged, qrels_set, all_doc_ids)
        all_binary.append((relevance, ranking))

        scores_g, ranking_g = results_to_ranking_graded(
            merged, qrels_graded.get(tid, {}), all_doc_ids
        )
        all_graded.append((scores_g, ranking_g))

        q_recalls, q_precs = pr_curve(ranking, relevance)

        per_query[tid] = {
            "AP":      average_precision(ranking, relevance),
            "RR":      reciprocal_rank(ranking, relevance),
            "P@10":    precision_at_k(ranking, relevance, 10),
            "R@100":   recall_at_k(ranking, relevance, 100),
            "NDCG@10": ndcg_at_k(ranking_g, scores_g, 10),
            "pr_curve": (q_recalls, q_precs),
        }

    rl, mp = mean_pr_curve(all_binary)
    return {
        "run":       run,
        "MAP":       mean_average_precision(all_binary),
        "MRR":       mean_reciprocal_rank(all_binary),
        "P@10":      float(np.mean([v["P@10"]    for v in per_query.values()])),
        "R@100":     float(np.mean([v["R@100"]   for v in per_query.values()])),
        "NDCG@10":   mean_ndcg_at_k(all_graded, k=10),
        "pr_curves": (rl, mp),
        "per_query": per_query,
    }


# ---------------------------------------------------------------------------
# Print comparison table
# ---------------------------------------------------------------------------

def print_comparison(results: dict[str, dict], label: str = "Test Set") -> None:
    """Print a clean comparison table for all strategies."""
    print(f"\n{'=' * 78}")
    print(f"Final Evaluation -- {label} (field={QUERY_FIELD})")
    print(f"{'=' * 78}")
    print(f"{'Strategy':>22} | {'MAP':>8} | {'MRR':>8} | {'P@10':>8} | {'R@100':>8} | {'NDCG@10':>9}")
    print("-" * 78)
    for name, r in results.items():
        print(
            f"{name:>22} | {r['MAP']:>8.4f} | {r['MRR']:>8.4f} | "
            f"{r['P@10']:>8.4f} | {r['R@100']:>8.4f} | {r['NDCG@10']:>9.4f}"
        )
    print("-" * 78)

    best_map = max(results, key=lambda n: results[n]["MAP"])
    print(f"\n  Best MAP: {best_map} ({results[best_map]['MAP']:.4f})")


# ---------------------------------------------------------------------------
# Convenience entry-point callable from notebooks or other scripts
# ---------------------------------------------------------------------------

def run_final_evaluation(
    client=None,
    index_name: str = "",
    corpus: list[dict] | None = None,
    test_topics: list[dict] | None = None,
    train_topics: list[dict] | None = None,
    qrels: dict | None = None,
    qrels_graded: dict | None = None,
    output_dir: str | Path = PHASE1_DIR,
    add_medcpt: bool = False,
) -> dict:
    """
    Run the full final Phase 1 evaluation (baseline + tuned, test set) and save run files.

    Always overwrites existing run files.

    Args:
        client:       OpenSearch client (created from env if None)
        index_name:   index to query (read from OPENSEARCH_INDEX env if empty)
        corpus:       list of corpus dicts (loaded from disk if None)
        test_topics:  test topic list (loaded from disk if None)
        train_topics: train topic list (loaded from disk if None)
        qrels:        binary qrels dict (loaded from disk if None)
        qrels_graded: graded qrels dict (loaded from disk if None)
        output_dir:   directory to write run JSON files
        add_medcpt:   index MedCPT embeddings into OpenSearch before evaluation

    Returns:
        dict with keys 'baseline_results' and 'tuned_results'
    """
    output_dir = Path(output_dir)

    from dotenv import load_dotenv
    load_dotenv()

    if corpus is None:
        corpus = load_corpus(ROOT / "data" / "filtered_pubmed_abstracts.txt")
    all_doc_ids = [doc["id"] for doc in corpus]

    if test_topics is None:
        with open(ROOT / "results" / "splits" / "test_queries.json") as f:
            test_topics = json.load(f)
    if train_topics is None:
        with open(ROOT / "results" / "splits" / "train_queries.json") as f:
            train_topics = json.load(f)
    if qrels is None:
        with open(ROOT / "results" / "qrels.json") as f:
            qrels = json.load(f)
    if qrels_graded is None:
        with open(ROOT / "results" / "qrels_graded.json") as f:
            qrels_graded = json.load(f)

    print(f"Corpus: {len(corpus)} docs  |  Test: {len(test_topics)} queries")

    if client is None:
        client = get_client()
    if not index_name:
        index_name = os.getenv("OPENSEARCH_INDEX", "")
    assert index_name, "OPENSEARCH_INDEX not set"

    if add_medcpt:
        add_medcpt_field(client, index_name)
        populate_medcpt_embeddings(client, index_name, corpus)

    from src.retrieval.lm_jelinek_mercer import LMJMRetriever
    from src.embeddings.encoder import Encoder
    from src.retrieval.knn import KNNRetriever

    lmjm       = LMJMRetriever(client, index_name, lambda_variant=LMJM_VARIANT)
    encoder    = Encoder()
    bm25_base  = FieldRetriever(client, index_name, "contents")
    lmdir_base = FieldRetriever(client, index_name, "contents_lmdir")
    knn_base   = KNNRetriever(client, index_name, encoder=encoder)
    bm25_tuned  = FieldRetriever(client, index_name, "contents_bm25_k15_b10")
    lmdir_tuned = FieldRetriever(client, index_name, "contents_lmdir_75")

    mapping   = client.indices.get_mapping(index=index_name)
    actual_key = list(mapping.keys())[0]
    has_medcpt = "embedding_medcpt" in mapping[actual_key]["mappings"]["properties"]

    print("=" * 78)
    print("Phase 1 -- Final Evaluation with Tuned Parameters")
    print("=" * 78)

    print("\n--- Evaluating BASELINE strategies on test set ---")
    baseline_results = {}
    for label, retriever in [
        ("BM25 (default)",   bm25_base),
        ("LM-JM (lam=0.7)", lmjm),
        ("LM-Dir (mu=2000)", lmdir_base),
        ("KNN (msmarco)",    knn_base),
    ]:
        print(f"  {label} ...", end="", flush=True)
        baseline_results[label] = evaluate_retriever(retriever, test_topics, qrels, qrels_graded, all_doc_ids)
        print(" done")

    print("  RRF (BM25+msmarco) ...", end="", flush=True)
    baseline_results["RRF (default)"] = evaluate_rrf(bm25_base, knn_base, test_topics, qrels, qrels_graded, all_doc_ids)
    print(" done")
    print_comparison(baseline_results, "Test Set -- BASELINE (default params)")

    print("\n--- Evaluating TUNED strategies on test set ---")
    tuned_results = {}

    print("  BM25 (k1=1.5, b=1.0) ...", end="", flush=True)
    tuned_results["BM25 (tuned)"] = evaluate_retriever(bm25_tuned, test_topics, qrels, qrels_graded, all_doc_ids)
    print(" done")

    tuned_results["LM-JM (lam=0.7)"] = baseline_results["LM-JM (lam=0.7)"]

    print("  LM-Dir (mu=75) ...", end="", flush=True)
    tuned_results["LM-Dir (mu=75)"] = evaluate_retriever(lmdir_tuned, test_topics, qrels, qrels_graded, all_doc_ids)
    print(" done")

    if has_medcpt:
        medcpt_knn = MedCPTKNNRetriever(client, index_name)
        print("  KNN (MedCPT) ...", end="", flush=True)
        tuned_results["KNN (MedCPT)"] = evaluate_retriever(medcpt_knn, test_topics, qrels, qrels_graded, all_doc_ids)
        print(" done")
        print("  RRF (tuned BM25 + MedCPT) ...", end="", flush=True)
        tuned_results["RRF (tuned)"] = evaluate_rrf(bm25_tuned, medcpt_knn, test_topics, qrels, qrels_graded, all_doc_ids)
        print(" done")
    else:
        print("  [skip] MedCPT KNN -- embedding_medcpt field not in index. Re-run with add_medcpt=True.")
        print("  RRF (tuned BM25 + msmarco) ...", end="", flush=True)
        tuned_results["RRF (tuned BM25)"] = evaluate_rrf(bm25_tuned, knn_base, test_topics, qrels, qrels_graded, all_doc_ids)
        print(" done")

    print_comparison(tuned_results, "Test Set -- TUNED (best params from train)")

    # save run files
    for name, r in tuned_results.items():
        safe = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        save_run(r["run"], output_dir / f"{safe}_run.json")

    summary = {n: {"MAP": r["MAP"], "MRR": r["MRR"], "P@10": r["P@10"],
                   "R@100": r["R@100"], "NDCG@10": r["NDCG@10"]}
               for n, r in {**baseline_results, **tuned_results}.items()}
    with open(output_dir / "final_eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {output_dir / 'final_eval_summary.json'}")

    return {"baseline_results": baseline_results, "tuned_results": tuned_results}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(name)s  %(message)s")

    parser = argparse.ArgumentParser(description="Final Phase 1 evaluation with tuned params")
    parser.add_argument("--add-medcpt", action="store_true",
                        help="Add MedCPT embedding field to index before evaluation")
    args = parser.parse_args()

    run_final_evaluation(add_medcpt=args.add_medcpt)
