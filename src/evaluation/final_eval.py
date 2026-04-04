"""
src/evaluation/final_eval.py

Phase 1 locked configuration constants and the run_final_evaluation() entry point.

Locked best configuration (from 5-fold CV on 32 train queries — see §3 of the notebook):
    query_field  = "topic+question"
    BM25         = k1=1.5, b=1.0
    LM-Dir       = mu=75
    LM-JM        = lambda=0.7
    Encoder      = MedCPT (ncbi/MedCPT-Query-Encoder + ncbi/MedCPT-Article-Encoder)

Public API:
    PHASE_1_BEST_CONFIG    — locked config dict
    run_final_evaluation() — run all strategies on test set, save run files
"""

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from src.data.loader import load_corpus
from src.evaluation.evaluator import evaluate_retriever, save_run
from src.indexing.index_builder import _BEST_PARAMS
from src.indexing.opensearch_client import get_client
from src.retrieval.knn import KNNRetriever, MedCPTKNNRetriever
from src.retrieval.rrf import RRFRetriever

logger = logging.getLogger(__name__)


######################################################################
## Locked Phase 1 configuration
## These come from 5-fold CV on 32 train queries (see results/phase1/tuning/)
######################################################################

_lmjm_best  = _BEST_PARAMS["lmjm_lambdas"][0]          # 0.7
_lmdir_best = _BEST_PARAMS["lmdir_mus"][0]              # 75
_bm25_k1    = _BEST_PARAMS["bm25_k1_b_pairs"][0][0]    # 1.5
_bm25_b     = _BEST_PARAMS["bm25_k1_b_pairs"][0][1]    # 1.0

PHASE_1_BEST_CONFIG = {
    "query_field":  "topic+question",
    "lmjm_variant": str(_lmjm_best).replace(".", ""),
    "lmjm_field":   f"contents_lmjm_{str(_lmjm_best).replace('.', '')}",
    "lmdir_field":  f"contents_lmdir_{_lmdir_best}",
    "bm25_field":   f"contents_bm25_k{str(_bm25_k1).replace('.','')}_b{str(_bm25_b).replace('.', '')}",
    "knn_field":    "embedding_medcpt",
    "encoder":      _BEST_PARAMS["encoders"][0][1],     # ncbi/MedCPT-Query-Encoder
}

QUERY_FIELD = PHASE_1_BEST_CONFIG["query_field"]
PHASE1_DIR  = ROOT / "results" / "phase1"


######################################################################
## Entry point
######################################################################

def run_final_evaluation(
    client=None,
    index_name:   str = "",
    corpus:       list[dict] | None = None,
    test_topics:  list[dict] | None = None,
    train_topics: list[dict] | None = None,
    qrels:        dict | None = None,
    qrels_graded: dict | None = None,
    output_dir:   str | Path = PHASE1_DIR,
) -> dict:
    """
    Run the full final Phase 1 evaluation (baseline + tuned on the test set) and
    save run files.  Always overwrites existing run files.

    Args:
        client:       OpenSearch client (created from .env if None)
        index_name:   index to query (read from OPENSEARCH_INDEX env if empty)
        corpus:       list of corpus dicts (loaded from disk if None)
        test_topics:  test topic list (loaded from disk if None)
        train_topics: train topic list (loaded from disk if None) — kept for API compat
        qrels:        binary qrels dict (loaded from disk if None)
        qrels_graded: graded qrels dict (loaded from disk if None)
        output_dir:   directory to write run JSON files

    Returns:
        {"baseline_results": dict, "tuned_results": dict}

    Note: index setup (adding MedCPT field, populating embeddings) is handled by
    create_or_update_index + index_documents in the notebook indexing cells, not here.
    """
    from src.retrieval.base import SparseRetriever
    from src.retrieval.lm_jelinek_mercer import LMJMRetriever
    from src.embeddings.encoder import Encoder

    output_dir = Path(output_dir)

    if corpus is None:
        corpus = load_corpus(ROOT / "data" / "filtered_pubmed_abstracts.txt")
    all_doc_ids = [doc["id"] for doc in corpus]

    if test_topics is None:
        with open(ROOT / "results" / "splits" / "test_queries.json") as f:
            test_topics = json.load(f)
    if qrels is None:
        with open(ROOT / "results" / "qrels" / "qrels.json") as f:
            qrels = json.load(f)
    if qrels_graded is None:
        with open(ROOT / "results" / "qrels" / "qrels_graded.json") as f:
            qrels_graded = json.load(f)

    print(f"Corpus: {len(corpus)} docs  |  Test: {len(test_topics)} queries")

    if client is None:
        client = get_client()
    if not index_name:
        index_name = os.getenv("OPENSEARCH_INDEX", "")
    assert index_name, "OPENSEARCH_INDEX not set"

    # check if MedCPT field is available in the index
    # (populated in §3.1 indexing cells via create_or_update_index + index_documents)
    mapping    = client.indices.get_mapping(index=index_name)
    actual_key = list(mapping.keys())[0]
    has_medcpt = "embedding_medcpt" in mapping[actual_key]["mappings"]["properties"]

    # ── Build retrievers ──────────────────────────────────────────────────────
    lmjm        = LMJMRetriever(client, index_name, lambd=_lmjm_best)
    encoder     = Encoder()
    bm25_base   = SparseRetriever(client, index_name, "contents_bm25_k12_b075")
    lmdir_base  = SparseRetriever(client, index_name, "contents_lmdir_2000")
    knn_base    = KNNRetriever(client, index_name, encoder=encoder)
    bm25_tuned  = SparseRetriever(client, index_name, "contents_bm25_k15_b10")
    lmdir_tuned = SparseRetriever(client, index_name, "contents_lmdir_75")

    def _eval(ret):
        return evaluate_retriever(ret, test_topics, qrels, qrels_graded, all_doc_ids,
                                  query_field=QUERY_FIELD)

    def _header(label):
        print(f"\n{'=' * 78}")
        print(f"  {label}")
        print(f"{'=' * 78}")

    # ── Baseline strategies ───────────────────────────────────────────────────
    _header("Baseline strategies — test set")
    baseline_results = {}
    for label, ret in [
        ("BM25 (default)",   bm25_base),
        ("LM-JM (lam=0.7)", lmjm),
        ("LM-Dir (mu=2000)", lmdir_base),
        ("KNN (msmarco)",    knn_base),
    ]:
        print(f"  {label} ...", end="", flush=True)
        baseline_results[label] = _eval(ret)
        print(" done")

    print("  RRF (BM25+msmarco) ...", end="", flush=True)
    baseline_results["RRF (default)"] = _eval(RRFRetriever(bm25_base, knn_base, rrf_k=60))
    print(" done")

    # ── Tuned strategies ──────────────────────────────────────────────────────
    _header("Tuned strategies — test set")
    tuned_results = {}

    print("  BM25 (k1=1.5, b=1.0) ...", end="", flush=True)
    tuned_results["BM25 (tuned)"] = _eval(bm25_tuned)
    print(" done")

    tuned_results["LM-JM (lam=0.7)"] = baseline_results["LM-JM (lam=0.7)"]

    print("  LM-Dir (mu=75) ...", end="", flush=True)
    tuned_results["LM-Dir (mu=75)"] = _eval(lmdir_tuned)
    print(" done")

    if has_medcpt:
        medcpt_knn = MedCPTKNNRetriever(client, index_name)
        print("  KNN (MedCPT) ...", end="", flush=True)
        tuned_results["KNN (MedCPT)"] = _eval(medcpt_knn)
        print(" done")
        print("  RRF (tuned BM25 + MedCPT) ...", end="", flush=True)
        tuned_results["RRF (tuned)"] = _eval(RRFRetriever(bm25_tuned, medcpt_knn, rrf_k=60))
        print(" done")
    else:
        print("  [skip] KNN (MedCPT) — 'embedding_medcpt' not in index (re-run with add_medcpt=True)")
        print("  RRF (tuned BM25 + msmarco) ...", end="", flush=True)
        tuned_results["RRF (tuned BM25)"] = _eval(RRFRetriever(bm25_tuned, knn_base, rrf_k=60))
        print(" done")

    # ── Print comparison table ────────────────────────────────────────────────
    for split_label, results in [
        ("BASELINE (default params)", baseline_results),
        ("TUNED (best params from train)", tuned_results),
    ]:
        print(f"\n{'=' * 78}")
        print(f"Final Evaluation -- Test Set ({split_label})")
        print(f"{'=' * 78}")
        print(f"{'Strategy':>24} | {'MAP':>8} | {'MRR':>8} | {'P@10':>8} | {'R@100':>8} | {'NDCG@100':>9}")
        print("-" * 78)
        for name, r in results.items():
            print(
                f"{name:>24} | {r['MAP']:>8.4f} | {r['MRR']:>8.4f} | "
                f"{r['P@10']:>8.4f} | {r['R@100']:>8.4f} | {r['NDCG@100']:>9.4f}"
            )

    # ── Save run files ────────────────────────────────────────────────────────
    for name, r in tuned_results.items():
        safe = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        save_run(r["run"], output_dir / f"{safe}_run.json")

    summary = {
        n: {"MAP": r["MAP"], "MRR": r["MRR"], "P@10": r["P@10"],
            "R@100": r["R@100"], "NDCG@100": r["NDCG@100"]}
        for n, r in {**baseline_results, **tuned_results}.items()
    }
    out_path = output_dir / "final_eval_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {out_path}")

    return {"baseline_results": baseline_results, "tuned_results": tuned_results}


#################################################################
##                  LOCAL TEST                                 ##
#################################################################
if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(name)s  %(message)s")
    run_final_evaluation()
