import csv
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from src.evaluation.evaluator import evaluate_retriever
from src.evaluation.metrics import (
    average_precision,
    mean_average_precision,
    mean_ndcg_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
    results_to_ranking,
)
from src.indexing.index_builder import float_tag, get_live_fields
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.lm_dirichlet import LMDirichletRetriever
from src.retrieval.lm_jelinek_mercer import LMJMRetriever
from src.tuning.cv_utils import run_cv

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SweepResult Class — returned by every run_*_sweep function
# Wraps the list-of-dicts returned by a hyperparameter sweep,
# So we can convert it to a DataFrame for better display
# ---------------------------------------------------------------------------

@dataclass
class SweepResult:
    rows:        list[dict]     # list of result dicts, already sorted by mean_ndcg descending
    param_col:   str            # name of the hyperparameter column (e.g. "mu", "lambda", "k1")
    kind:        str            # model type: "lmdir" | "lmjm" | "bm25" | "encoder"
    baseline_id: Any            # values of the baseline params of that model
    meta:        dict = field(default_factory=dict)  # optional extra context (n_folds, query_field, …)


    # param label to display in the df for each param_col
    _PARAM_LABELS: ClassVar[dict] = {
        "mu":     "μ",
        "lambda": "λ",
        "k1":     "k1",
        "b":      "b",
        "alias":  "Encoder",
    }

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return DataFrame sorted by param value
        Columns depend on kind: lmdir/lmjm : μ/λ; BM25: k1, b; encoder: alias; rrf: label
        Common colums: NDCG@100 | ±std | MAP | MRR | ΔNDCG
        """
        if self.kind == "bm25":
            return self._df_bm25()
        if self.kind == "encoder":
            return self._df_encoder()
        if self.kind == "rrf":
            return self._df_rrf()
        return self._df_1d()

    def _df_1d(self) -> pd.DataFrame:
        param       = self.param_col
        col_label   = self._PARAM_LABELS.get(param, param)
        base_r      = self.baseline()
        base_v      = base_r["mean_ndcg"] if base_r else 0.0
        records = []
        for r in sorted(self.rows, key=lambda x: x[param]):
            is_base = (r[param] == self.baseline_id)
            delta   = r["mean_ndcg"] - base_v
            records.append({
                col_label:   r[param],
                "NDCG@100":   round(r["mean_ndcg"], 4),
                "±std":      round(r["std_ndcg"],  4),
                "MAP":       round(r["mean_map"],   4),
                "MRR":       round(r["mean_mrr"],   4),
                "ΔNDCG":     "baseline" if is_base else f"{delta:+.4f}",
            })
        return pd.DataFrame(records)

    def _df_bm25(self) -> pd.DataFrame:
        base_r  = self.baseline()
        base_v  = base_r["mean_ndcg"] if base_r else 0.0
        records = []
        for r in sorted(self.rows, key=lambda x: (x["k1"], x["b"])):
            is_base = (r["k1"], r["b"]) == self.baseline_id
            delta   = r["mean_ndcg"] - base_v
            records.append({
                "k1":      r["k1"],
                "b":       r["b"],
                "NDCG@100": round(r["mean_ndcg"], 4),
                "±std":    round(r["std_ndcg"],  4),
                "MAP":     round(r["mean_map"],   4),
                "ΔNDCG":   "baseline" if is_base else f"{delta:+.4f}",
            })
        return pd.DataFrame(records)

    def _df_encoder(self) -> pd.DataFrame:
        base_r = self.baseline()
        base_v = base_r["ndcg"] if base_r else 0.0
        records = []
        for r in sorted(self.rows, key=lambda x: x["ndcg"], reverse=True):
            is_base = (r["alias"] == self.baseline_id)
            delta   = r["ndcg"] - base_v
            records.append({
                "Encoder": r["alias"],
                "NDCG@100": round(r["ndcg"], 4),
                "MAP":     round(r["map"],   4),
                "MRR":     round(r["mrr"],   4),
                "P@10":    round(r["p10"],   4),
                "ΔNDCG":   "baseline" if is_base else f"{delta:+.4f}",
            })
        return pd.DataFrame(records)

    def _df_rrf(self) -> pd.DataFrame:
        base_r = self.baseline()
        base_v = base_r["mean_ndcg"] if base_r else 0.0
        records = []
        for r in sorted(self.rows, key=lambda x: x["mean_ndcg"], reverse=True):
            is_base = (r["label"] == self.baseline_id)
            delta   = r["mean_ndcg"] - base_v
            records.append({
                "Pair":      r["pair"],
                "RRF k":     r["rrf_k"],
                "NDCG@100":  round(r["mean_ndcg"], 4),
                "±std":      round(r["std_ndcg"],  4),
                "MAP":       round(r["mean_map"],   4),
                "MRR":       round(r["mean_mrr"],   4),
                "P@10":      round(r["mean_p10"],   4),
                "ΔNDCG":     "baseline" if is_base else f"{delta:+.4f}",
            })
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Gtters
    # ------------------------------------------------------------------
    @property
    def best(self) -> dict:
        """return row with highest NDCG@100 (rows already sorted descending)."""
        return self.rows[0]

    def baseline(self) -> dict | None:
        """Row matching baseline_id, or None if not found."""
        if self.kind == "bm25":
            k1_b = self.baseline_id          # (k1, b) tuple
            return next(
                (r for r in self.rows if r["k1"] == k1_b[0] and r["b"] == k1_b[1]),
                None,
            )
        if self.kind == "encoder":
            return next(
                (r for r in self.rows if r["alias"] == self.baseline_id),
                None,
            )
        if self.kind == "rrf":
            # baseline_id is the pair label (without k suffix); match lowest rrf_k row
            candidates = [r for r in self.rows if r["pair"] == self.baseline_id]
            if not candidates:
                return None
            return min(candidates, key=lambda r: r["rrf_k"])
        # 1-D sweeps: param_col value
        return next(
            (r for r in self.rows if r[self.param_col] == self.baseline_id),
            None,
        )

    def __repr__(self) -> str:
        """Summary of sweep showing best hyperparam(s) and NDCG@100."""
        b = self.best
        if self.kind == "bm25":
            best_str = f"k1={b['k1']}, b={b['b']}"
            ndcg_val = b["mean_ndcg"]
        elif self.kind == "encoder":
            best_str = b["alias"]
            ndcg_val = b["ndcg"]
        elif self.kind == "rrf":
            best_str = b["label"]
            ndcg_val = b["mean_ndcg"]
        else:
            best_str = f"{self.param_col}={b[self.param_col]}"
            ndcg_val = b["mean_ndcg"]
        return (
            f"SweepResult(kind={self.kind!r}, n={len(self.rows)}, "
            f"best={best_str}, NDCG@100={ndcg_val:.4f})"
        )


######################################################################
## Unified hyperparameter sweeper
##
## Primary selection criterion: NDCG@100 (graded qrels).
## MAP is computed and shown alongside but is NOT the sort key.
##
######################################################################


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_fields_exist(client, index_name: str, fields: list[str]) -> None:
    """Raise ValueError if any required fields are missing from the live index."""
    live = get_live_fields(client, index_name)
    missing = [f for f in fields if f not in live]
    if missing:
        raise ValueError(
            f"[sweeper] Required field(s) not in index '{index_name}': {missing}\n"
            "  -> Run the index build cell (section 3.1) first."
        )


def _save_csv(results: list[dict], path: Path) -> None:
    """Write results list to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"[sweeper] Saved -> {path}")


def _fold_row(cv: dict) -> dict:
    """Per-fold NDCG columns (up to 5) for the CSV."""
    return {
        f"ndcg_fold_{i+1}": cv["ndcg_per_fold"][i] if i < len(cv["ndcg_per_fold"]) else None
        for i in range(5)
    }



# ---------------------------------------------------------------------------
# Query field ablation (cheap, single-run on train set)
# ---------------------------------------------------------------------------

def field_ablation(
    client=None,
    index_name: str = "",
    all_doc_ids: list[str] = None,
    train_topics: list[dict] = None,
    qrels: dict[str, dict] = None,
    qrels_graded: dict[str, dict] = None,
    retriever=None,
) -> str:
    """
    Run a retriever with 6 query field variants on the train set, pick the one
    with highest NDCG@100.

    Fields tested:
        'topic'           -- short keyword label (3-8 words)
        'question'        -- full clinical question (10-20 words)
        'narrative'       -- extended relevance description (30-60 words)
        'topic+question'  -- topic + question (no narrative)
        'topic+narrative' -- topic + narrative (no question)
        'concatenated'    -- topic + question + narrative (all three)

    Args:
        client:       OpenSearch client (required when retriever is None)
        index_name:   OpenSearch index name (required when retriever is None)
        all_doc_ids:  full corpus doc-ID list
        train_topics: list of train topic dicts
        qrels:        binary qrels {topic_id: {pmid: 1}}
        qrels_graded: graded qrels {topic_id: {pmid: 0/1/2}}
        retriever:    (optional) a pre-built retriever instance.  When provided,
                      client and index_name are ignored and this retriever is used
                      directly for all 6 field variants.  Useful for ablating any
                      strategy (e.g. RRF best) rather than only BM25 default.

    Returns (winner: str, results: dict)
    """
    if retriever is None:
        if client is None or not index_name:
            raise ValueError(
                "field_ablation: either supply a pre-built retriever instance, "
                "or both `client` and `index_name`."
            )
        retriever = BM25Retriever(client, index_name)

    fields = ["topic", "question", "narrative", "topic+question", "topic+narrative", "concatenated"]
    results = {}

    for field in fields:
        r = evaluate_retriever(retriever, train_topics, qrels, qrels_graded, all_doc_ids, query_field=field)
        results[field] = r

    # Pick winner by NDCG@100 (graded relevance) — MAP shown for reference only
    winner = max(results, key=lambda f: results[f]["NDCG@100"])

    return winner, results


# ---------------------------------------------------------------------------
# BM25 (k1, b) 2-D grid sweep
# ---------------------------------------------------------------------------

def run_bm25_sweep(
    client,
    index_name:   str,
    train_topics: list[dict],
    qrels:        dict,
    qrels_graded: dict,
    all_doc_ids:  list[str],
    k1_b_grid:    list[tuple[float, float]],
    n_folds:      int = 5,
    query_field:  str = "concatenated",
    output_csv:   Path | None = None,
) -> "SweepResult":
    """
    BM25 (k1, b) grid sweep via k-fold CV. Sorted by mean NDCG@100 (primary criterion).

    Args:
        k1_b_grid:  list of (k1, b) tuples from BM25_K1_B_GRID constant
        output_csv: if given, save results to this path

    Returns:
        SweepResult  (kind="bm25", baseline_id=(1.2, 0.75))
    """
    required = [f"contents_bm25_k{float_tag(k1)}_b{float_tag(b)}" for k1, b in k1_b_grid]
    _assert_fields_exist(client, index_name, required)

    results = []
    n = len(k1_b_grid)
    print(f"\nBM25 (k1, b) sweep -- {n} configs x {n_folds} folds  [sorted by NDCG@100]")
    for i, (k1, b) in enumerate(k1_b_grid, 1):
        factory = lambda k1=k1, b=b: BM25Retriever(client, index_name, k1=k1, b=b)
        cv = run_cv(
            factory, train_topics, qrels, qrels_graded, all_doc_ids,
            query_field=query_field, n_folds=n_folds, verbose=False,
        )
        results.append({
            "k1": k1, "b": b,
            "mean_ndcg": cv["mean_ndcg"], "std_ndcg": cv["std_ndcg"],
            "mean_map":  cv["mean_map"],  "std_map":  cv["std_map"],
            "mean_mrr":  cv["mean_mrr"],  "mean_p10": cv["mean_p10"],
            **_fold_row(cv),
        })

    results.sort(key=lambda r: r["mean_ndcg"], reverse=True)
    best = results[0]
    print(f"  >> Best: k1={best['k1']}, b={best['b']}  "
          f"NDCG@100={best['mean_ndcg']:.4f}  MAP={best['mean_map']:.4f}")

    if output_csv:
        _save_csv(results, Path(output_csv))

    return SweepResult(
        rows        = results,
        param_col   = "k1",       # primary axis; b is secondary
        kind        = "bm25",
        baseline_id = (1.2, 0.75),
        meta        = {"n_folds": n_folds, "query_field": query_field},
    )


# ---------------------------------------------------------------------------
# LM Jelinek-Mercer lambda sweep
# ---------------------------------------------------------------------------

def run_lmjm_sweep(
    client,
    index_name:   str,
    train_topics: list[dict],
    qrels:        dict,
    qrels_graded: dict,
    all_doc_ids:  list[str],
    lambdas:      list[float],
    n_folds:      int = 5,
    query_field:  str = "concatenated",
    output_csv:   Path | None = None,
) -> "SweepResult":
    """
    LM Jelinek-Mercer lambda sweep via k-fold CV. Sorted by mean NDCG@100.

    Returns:
        SweepResult  (kind="lmjm", baseline_id=0.7)
    """
    required = [f"contents_lmjm_{float_tag(lam)}" for lam in lambdas]
    _assert_fields_exist(client, index_name, required)

    results = []
    print(f"\nLM-JM lambda sweep -- {len(lambdas)} values x {n_folds} folds  [sorted by NDCG@100]")
    for lam in lambdas:
        factory = lambda lam=lam: LMJMRetriever(client, index_name, lambd=lam)
        cv = run_cv(
            factory, train_topics, qrels, qrels_graded, all_doc_ids,
            query_field=query_field, n_folds=n_folds, verbose=False,
        )
        results.append({
            "lambda": lam,
            "mean_ndcg": cv["mean_ndcg"], "std_ndcg": cv["std_ndcg"],
            "mean_map":  cv["mean_map"],  "std_map":  cv["std_map"],
            "mean_mrr":  cv["mean_mrr"],  "mean_p10": cv["mean_p10"],
            **_fold_row(cv),
        })

    results.sort(key=lambda r: r["mean_ndcg"], reverse=True)
    best = results[0]
    print(f"  >> Best: lambda={best['lambda']:.4f}  "
          f"NDCG@100={best['mean_ndcg']:.4f}  MAP={best['mean_map']:.4f}")

    if output_csv:
        _save_csv(results, Path(output_csv))

    return SweepResult(
        rows        = results,
        param_col   = "lambda",
        kind        = "lmjm",
        baseline_id = 0.7,
        meta        = {"n_folds": n_folds, "query_field": query_field},
    )


# ---------------------------------------------------------------------------
# LM Dirichlet mu sweep
# ---------------------------------------------------------------------------

def run_lmdir_sweep(
    client,
    index_name:   str,
    train_topics: list[dict],
    qrels:        dict,
    qrels_graded: dict,
    all_doc_ids:  list[str],
    mus:          list[int],
    n_folds:      int = 5,
    query_field:  str = "concatenated",
    output_csv:   Path | None = None,
) -> "SweepResult":
    """
    LM Dirichlet mu sweep via k-fold CV. Sorted by mean NDCG@100.

    Returns:
        SweepResult  (kind="lmdir", baseline_id=2000)
    """
    required = [f"contents_lmdir_{mu}" for mu in mus]
    _assert_fields_exist(client, index_name, required)

    results = []
    print(f"\nLM-Dir mu sweep -- {len(mus)} values x {n_folds} folds  [sorted by NDCG@100]")
    for mu in mus:
        factory = lambda mu=mu: LMDirichletRetriever(client, index_name, mu=mu)
        cv = run_cv(
            factory, train_topics, qrels, qrels_graded, all_doc_ids,
            query_field=query_field, n_folds=n_folds, verbose=False,
        )
        results.append({
            "mu": mu,
            "mean_ndcg": cv["mean_ndcg"], "std_ndcg": cv["std_ndcg"],
            "mean_map":  cv["mean_map"],  "std_map":  cv["std_map"],
            "mean_mrr":  cv["mean_mrr"],  "mean_p10": cv["mean_p10"],
            **_fold_row(cv),
        })

    results.sort(key=lambda r: r["mean_ndcg"], reverse=True)
    best = results[0]
    print(f"  >> Best: mu={best['mu']}  "
          f"NDCG@100={best['mean_ndcg']:.4f}  MAP={best['mean_map']:.4f}")

    if output_csv:
        _save_csv(results, Path(output_csv))

    return SweepResult(
        rows        = results,
        param_col   = "mu",
        kind        = "lmdir",
        baseline_id = 2000,
        meta        = {"n_folds": n_folds, "query_field": query_field},
    )


# ---------------------------------------------------------------------------
# Dense encoder comparison (exact cosine, no OpenSearch)
# ---------------------------------------------------------------------------

def _eval_encoder_exact_cosine(
    doc_embs:     np.ndarray,
    query_embs:   np.ndarray,
    topics:       list[dict],
    qrels:        dict,
    qrels_graded: dict,
    all_doc_ids:  list[str],
    top_k:        int = 100,
) -> dict:
    """
    Evaluate one encoder via brute-force cosine similarity. Primary metric: NDCG@100.

    Both matrices must be L2-normalised (dot product == cosine similarity).
    Returns {"NDCG@100": float, "MAP": float, "MRR": float, "P@10": float}.
    """
    scores_matrix = query_embs @ doc_embs.T          # (N_queries, N_docs)
    top_k         = min(top_k, doc_embs.shape[0])

    all_binary = []
    all_graded = []
    per_query  = {}

    for q_idx, topic in enumerate(topics):
        tid       = str(topic["id"])
        qrels_set = set(qrels.get(tid, {}).keys())

        scores    = scores_matrix[q_idx]
        top_idx   = np.argpartition(scores, -top_k)[-top_k:]
        top_idx   = top_idx[np.argsort(scores[top_idx])[::-1]]  # descending

        ranking   = top_idx.tolist()
        relevance = [all_doc_ids[i] in qrels_set for i in range(len(all_doc_ids))]
        all_binary.append((relevance, ranking))

        # graded path for NDCG
        scores_g = [float(qrels_graded.get(tid, {}).get(all_doc_ids[i], 0))
                    for i in range(len(all_doc_ids))]
        all_graded.append((scores_g, ranking))

        per_query[tid] = {
            "NDCG@100": ndcg_at_k(ranking, scores_g, 100),
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


def run_encoder_sweep(
    train_topics:    list[dict],
    qrels:           dict,
    qrels_graded:    dict,
    all_doc_ids:     list[str],
    embeddings_list: list[tuple],          # (alias, model_name, doc_embs) from corpus_encoder
    query_field:     str = "concatenated", # which field to use as query text
    cache_dir:       Path | None = None,   # directory to cache query embeddings (.npy files)
    force_reencode:  bool = False,
    batch_size:      int = 32,
    output_csv:      Path | None = None,
) -> "SweepResult":
    """
    Compare dense encoders via exact cosine on the full train set. Sorted by NDCG@100.

    Accepts the ``embeddings_list`` produced by ``create_embeddings()`` — each entry is
    ``(alias, model_name, doc_embs)``.  Query embeddings are encoded on-the-fly (or loaded
    from ``cache_dir`` if previously saved).

    Args:
        embeddings_list: list of (alias, model_name, doc_embs) — doc_embs must be L2-normalised
        query_field:     which topic field to use as query text ("topic"|"question"|"concatenated")
        cache_dir:       if given, query embeddings are cached/loaded from here
        force_reencode:  if True, re-encode even when a cache file exists
        batch_size:      encoding batch size
        output_csv:      if given, save results

    Returns:
        SweepResult  (kind="encoder", baseline_id="msmarco")
    """
    from src.embeddings.encoder import Encoder
    from src.data.query_builder import build_query

    # MedCPT uses an asymmetric architecture: separate query encoder
    _QUERY_MODEL_OVERRIDES = {
        "medcpt": "ncbi/MedCPT-Query-Encoder",
    }

    assert embeddings_list, "embeddings_list is empty"
    query_texts = [build_query(t, query_field) for t in train_topics]

    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # Build (alias, doc_embs, query_embs) triples
    encoder_specs = []
    for alias, model_name, doc_embs in embeddings_list:
        query_model = _QUERY_MODEL_OVERRIDES.get(alias, model_name)

        if cache_dir:
            qcache = Path(cache_dir) / f"{alias}_queries_{query_field}.npy"
            if not force_reencode and qcache.exists():
                query_embs = np.load(qcache)
                if query_embs.shape[0] > len(query_texts):
                    query_embs = query_embs[:len(query_texts)]
                print(f"[{alias}] Query embeddings loaded from cache  shape={query_embs.shape}")
            else:
                print(f"[{alias}] Encoding {len(query_texts)} queries with '{query_model}' ...")
                query_embs = Encoder(query_model).encode(query_texts, batch_size=batch_size)
                np.save(qcache, query_embs)
                print(f"[{alias}] Cached → {qcache.name}  shape={query_embs.shape}")
        else:
            print(f"[{alias}] Encoding {len(query_texts)} queries with '{query_model}' ...")
            query_embs = Encoder(query_model).encode(query_texts, batch_size=batch_size)

        de = doc_embs[:len(all_doc_ids)]   # guard: trim to actual corpus size
        encoder_specs.append((alias, de, query_embs))

    results = []
    print(f"Encoder sweep -- {len(encoder_specs)} encoder(s)  [sorted by NDCG@100]")
    for alias, doc_embs, query_embs in encoder_specs:
        assert doc_embs.shape[0] == len(all_doc_ids), (
            f"[{alias}] doc_embs rows {doc_embs.shape[0]} != corpus size {len(all_doc_ids)}"
        )
        assert query_embs.shape[0] == len(train_topics), (
            f"[{alias}] query_embs rows {query_embs.shape[0]} != train topics {len(train_topics)}"
        )
        m = _eval_encoder_exact_cosine(
            doc_embs, query_embs, train_topics, qrels, qrels_graded, all_doc_ids,
        )
        results.append({
            "alias":  alias,
            "ndcg":   m["NDCG@100"],
            "map":    m["MAP"],
            "mrr":    m["MRR"],
            "p10":    m["P@10"],
        })

    results.sort(key=lambda r: r["ndcg"], reverse=True)
    best = results[0]
    print(f"  >> Best: {best['alias']}  NDCG@100={best['ndcg']:.4f}  MAP={best['map']:.4f}")

    if output_csv:
        _save_csv(results, Path(output_csv))

    return SweepResult(
        rows        = results,
        param_col   = "alias",
        kind        = "encoder",
        baseline_id = "msmarco",
        meta        = {"query_field": query_field},
    )


# ---------------------------------------------------------------------------
# RRF pair grid search
# ---------------------------------------------------------------------------

def run_rrf_sweep(
    client,
    index_name:   str,
    train_topics: list[dict],
    qrels:        dict,
    qrels_graded: dict,
    all_doc_ids:  list[str],
    pair_configs: list[dict],
    rrf_k_grid:   list[int]  = None,
    n_folds:      int        = 5,
    query_field:  str        = "topic+question",
    solo_scores:  dict | None = None,   # {"BM25": {"ndcg":0.80, "map":…, "mrr":…, "p10":…}, …}
    output_csv:   Path | None = None,
) -> "SweepResult":
    """
    Grid search over RRF model pairs and the RRF smoothing constant k.

    Each element of pair_configs is a dict with keys:
        label   : str   — human-readable pair name, e.g. "BM25+KNN"
        factory : callable() → (retriever_a, retriever_b)

    rrf_k_grid controls the k values swept (default: [30, 60, 90]).

    solo_scores (optional): dict mapping model name → {"ndcg", "map", "mrr", "p10"}
        with the CV NDCG@100 of each retriever run solo.  Stored in SweepResult.meta
        and used by plot_rrf_sweep() to draw reference lines so you can see whether
        fusion beats any individual retriever.

    Sorted by mean NDCG@100. Returns SweepResult with kind="rrf".
    """
    from src.retrieval.rrf import RRFRetriever

    if rrf_k_grid is None:
        rrf_k_grid = [30, 60, 90]

    n_pairs = len(pair_configs)
    n_k     = len(rrf_k_grid)
    baseline_label = pair_configs[0]["label"]

    print(f"\nRRF pair sweep -- {n_pairs} pairs × {n_k} k-values × {n_folds} folds  "
          f"[sorted by NDCG@100]")

    results = []
    for pc in pair_configs:
        pair_label  = pc["label"]
        pair_factory = pc["factory"]   # () → (retriever_a, retriever_b)

        for rrf_k in rrf_k_grid:
            label = f"{pair_label}  k={rrf_k}"

            def _make_rrf(pf=pair_factory, k=rrf_k):
                ra, rb = pf()
                return RRFRetriever(ra, rb, rrf_k=k)

            cv = run_cv(
                _make_rrf, train_topics, qrels, qrels_graded, all_doc_ids,
                query_field=query_field, n_folds=n_folds, verbose=False,
            )
            results.append({
                "label":      label,
                "pair":       pair_label,
                "rrf_k":      rrf_k,
                "mean_ndcg":  cv["mean_ndcg"],
                "std_ndcg":   cv["std_ndcg"],
                "mean_map":   cv["mean_map"],
                "std_map":    cv["std_map"],
                "mean_mrr":   cv["mean_mrr"],
                "mean_p10":   cv["mean_p10"],
                **_fold_row(cv),
            })
            print(f"  {label:<32}  NDCG@100={cv['mean_ndcg']:.4f}  "
                  f"MAP={cv['mean_map']:.4f}")

    results.sort(key=lambda r: r["mean_ndcg"], reverse=True)
    best = results[0]
    print(f"\n  >> Best: '{best['label']}'  "
          f"NDCG@100={best['mean_ndcg']:.4f}  MAP={best['mean_map']:.4f}")

    if output_csv:
        _save_csv(results, Path(output_csv))

    return SweepResult(
        rows        = results,
        param_col   = "label",
        kind        = "rrf",
        baseline_id = baseline_label,
        meta        = {
            "n_folds":     n_folds,
            "query_field": query_field,
            "rrf_k_grid":  rrf_k_grid,
            "solo_scores": solo_scores or {},
        },
    )