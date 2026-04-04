"""
src/tuning/__sweeper_test.py

Comprehensive tests for cv_utils.py, sweeper.py, and tuning_plots.py.

Covers:
  1. make_folds()           — split counts, no overlap, reproducible, edge cases
  2. run_cv()               — offline mock retriever (no OpenSearch needed)
  3. SweepResult unit tests — to_dataframe(), best, baseline(), __repr__  (offline)
  4. Plot smoke tests       — plot_lmdir/lmjm/bm25/encoder_sweep()        (offline)
  5. run_bm25_sweep()       — 2-config grid on a real test index (integration)
  6. run_lmjm_sweep()       — 2-lambda grid on a real test index (integration)
  7. run_lmdir_sweep()      — 2-mu grid on a real test index (integration)
  8. run_encoder_sweep()    — exact-cosine eval with toy pre-built embeddings (offline)
  9. field_ablation()       — 3 query fields on a real test index (integration)

Tests 1-4, 8  : offline, no OpenSearch required — always run.
Tests 5-7, 9  : integration, require a live OpenSearch connection.
                Use a DEDICATED TEST INDEX (never touch production).
                Pattern mirrors __index_builder_test.py exactly.

Usage:
    cd C:\\Users\\franc\\Desktop\\NLP_Biomedical_Agent
    C:/Users/franc/anaconda3/envs/cnn/python.exe -m src.tuning.__sweeper_test
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
load_dotenv(ROOT / ".env")

logging.basicConfig(level=logging.WARNING, format="%(message)s")

from src.tuning.cv_utils import make_folds, run_cv
from src.tuning.sweeper import (
    SweepResult,
    _eval_encoder_exact_cosine,
    field_ablation,
    run_bm25_sweep,
    run_encoder_sweep,
    run_lmdir_sweep,
    run_lmjm_sweep,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _fake_topics(n: int) -> list[dict]:
    """Minimal topic dicts sufficient for CV splitting and retrieval tests."""
    return [{"id": str(i), "topic": f"topic {i}", "question": f"question {i}", "narrative": ""} for i in range(n)]


def _fake_qrels(topics: list[dict], doc_ids: list[str], n_rel: int = 2) -> dict:
    """Assign the first n_rel doc_ids as relevant for every topic."""
    return {t["id"]: {d: 1 for d in doc_ids[:n_rel]} for t in topics}


class _MockRetriever:
    """Returns the first `size` doc_ids with scores — no network call."""
    def __init__(self, doc_ids: list[str]):
        self._doc_ids = doc_ids

    def search(self, query: str, size: int = 100) -> list[tuple[str, float]]:
        top = self._doc_ids[:size]
        return [(d, 1.0 - i * 0.01) for i, d in enumerate(top)]


def _ok(msg: str) -> None:
    print(f"  [ok]  {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL]  {msg}")
    sys.exit(1)


def _assert(cond: bool, msg: str) -> None:
    if cond:
        _ok(msg)
    else:
        _fail(msg)


def _section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Integration test setup
#
# Pattern mirrors __index_builder_test.py exactly:
#   - use a dedicated test index so we never touch production
#   - create_or_update_index() with only the fields the sweep tests need
#   - index a small mini corpus (10 docs) into it
#   - run sweep tests against it
#   - delete the test index at the end
# ---------------------------------------------------------------------------

# Mini corpus: 10 synthetic docs, same structure as real corpus
MINI_CORPUS = [
    {"id": f"SWEEP_TEST_{i:04d}",
     "contents": f"This is synthetic abstract {i} about biomedical topic {i}. "
                 f"It contains medical terminology and clinical observations."}
    for i in range(10)
]
MINI_DOC_IDS = [d["id"] for d in MINI_CORPUS]
EMB_DIM = 4   # tiny, cheap — no real model loaded

# BM25 k1/b pairs to use in the integration tests
TEST_BM25_GRID = [(1.2, 0.75), (1.5, 1.0)]
TEST_LAMBDAS   = [0.1, 0.7]
TEST_MUS       = [75, 2000]


def _connect_opensearch():
    """
    Connect to OpenSearch exactly as __index_builder_test.py does.
    Uses get_client() + a lightweight indices.exists() probe (not ping()).
    Returns client, or prints [SKIP] and returns None.
    """
    from src.indexing.opensearch_client import get_client

    index_name = os.getenv("OPENSEARCH_INDEX", "")
    if not index_name:
        print("  [SKIP] OPENSEARCH_INDEX not set in .env — skipping integration tests")
        return None, None

    try:
        client = get_client()
    except Exception as exc:
        print(f"  [SKIP] get_client() failed ({exc}) — skipping integration tests")
        return None, None

    # lightweight connectivity probe — same as __index_builder_test.py
    # ping() returns False on 403 (permission denied, not unreachable)
    # indices.exists() on a harmless name gives a 403/404, both mean "server is up"
    try:
        client.indices.exists(index="probe-connection-check")
    except Exception as exc:
        err = str(exc)
        if "403" not in err and "404" not in err and "security_exception" not in err:
            print(f"  [SKIP] OpenSearch not reachable ({exc}) — skipping integration tests")
            return None, None

    test_index = f"{index_name}_sweeper_test"
    return client, test_index


def _build_test_index(client, test_index: str) -> None:
    """
    Create a fresh test index with all field variants needed for the sweep tests.
    Uses create_or_update_index() from index_builder — exactly as __index_builder_test.py.
    """
    from src.indexing.index_builder import IndexSettings, create_or_update_index, delete_index

    # clean slate
    delete_index(client, test_index)

    test_settings = IndexSettings(
        n_shards=1, n_replicas=0,
        refresh_interval="1s",
        ef_search=10, ef_construct=16, hnsw_m=4,
    )
    # create with ALL fields the sweep tests will need
    create_or_update_index(
        client, test_index,
        bm25_k1_b_pairs = TEST_BM25_GRID,
        lmjm_lambdas    = TEST_LAMBDAS,
        lmdir_mus       = TEST_MUS,
        encoders        = [],          # no KNN fields needed for sparse sweeps
        settings        = test_settings,
    )

    # index the mini corpus — all text fields get the same contents, that's fine
    from src.indexing.document_indexer import index_documents
    index_documents(client, test_index, MINI_CORPUS, {})
    client.indices.refresh(index=test_index)
    print(f"  [ok]  Test index '{test_index}' ready — {len(MINI_CORPUS)} docs")


def _delete_test_index(client, test_index: str) -> None:
    from src.indexing.index_builder import delete_index
    delete_index(client, test_index)
    print(f"  [ok]  Test index '{test_index}' deleted")


# ---------------------------------------------------------------------------
# Helpers to build synthetic SweepResult objects (offline)
# ---------------------------------------------------------------------------

def _make_lmdir_result() -> SweepResult:
    rows = [
        {"mu": 75,   "mean_ndcg": 0.62, "std_ndcg": 0.05, "mean_map": 0.55, "std_map": 0.04, "mean_mrr": 0.80, "mean_p10": 0.62},
        {"mu": 200,  "mean_ndcg": 0.59, "std_ndcg": 0.06, "mean_map": 0.53, "std_map": 0.05, "mean_mrr": 0.78, "mean_p10": 0.59},
        {"mu": 2000, "mean_ndcg": 0.51, "std_ndcg": 0.07, "mean_map": 0.46, "std_map": 0.06, "mean_mrr": 0.72, "mean_p10": 0.51},
    ]
    # rows sorted desc by ndcg (sweeper contract)
    return SweepResult(rows=rows, param_col="mu", kind="lmdir", baseline_id=2000,
                       meta={"n_folds": 5, "query_field": "concatenated"})


def _make_lmjm_result() -> SweepResult:
    rows = [
        {"lambda": 0.7, "mean_ndcg": 0.65, "std_ndcg": 0.04, "mean_map": 0.57, "std_map": 0.03, "mean_mrr": 0.83, "mean_p10": 0.65},
        {"lambda": 0.1, "mean_ndcg": 0.48, "std_ndcg": 0.08, "mean_map": 0.41, "std_map": 0.07, "mean_mrr": 0.68, "mean_p10": 0.48},
    ]
    return SweepResult(rows=rows, param_col="lambda", kind="lmjm", baseline_id=0.7,
                       meta={"n_folds": 5, "query_field": "concatenated"})


def _make_bm25_result() -> SweepResult:
    rows = [
        {"k1": 1.5, "b": 0.75, "mean_ndcg": 0.68, "std_ndcg": 0.04, "mean_map": 0.59, "std_map": 0.03, "mean_mrr": 0.85, "mean_p10": 0.68},
        {"k1": 1.2, "b": 0.75, "mean_ndcg": 0.67, "std_ndcg": 0.05, "mean_map": 0.57, "std_map": 0.04, "mean_mrr": 0.83, "mean_p10": 0.67},
        {"k1": 1.0, "b": 0.50, "mean_ndcg": 0.64, "std_ndcg": 0.06, "mean_map": 0.54, "std_map": 0.05, "mean_mrr": 0.80, "mean_p10": 0.64},
        {"k1": 0.5, "b": 0.25, "mean_ndcg": 0.60, "std_ndcg": 0.07, "mean_map": 0.51, "std_map": 0.06, "mean_mrr": 0.76, "mean_p10": 0.60},
    ]
    return SweepResult(rows=rows, param_col="k1", kind="bm25", baseline_id=(1.2, 0.75),
                       meta={"n_folds": 5, "query_field": "concatenated"})


def _make_encoder_result() -> SweepResult:
    rows = [
        {"alias": "medcpt",   "ndcg": 0.71, "map": 0.62, "mrr": 0.88, "p10": 0.72},
        {"alias": "multi-qa", "ndcg": 0.65, "map": 0.55, "mrr": 0.83, "p10": 0.65},
        {"alias": "msmarco",  "ndcg": 0.58, "map": 0.48, "mrr": 0.78, "p10": 0.58},
    ]
    return SweepResult(rows=rows, param_col="alias", kind="encoder", baseline_id="msmarco", meta={})


# ---------------------------------------------------------------------------
# 1. make_folds (offline)
# ---------------------------------------------------------------------------

def test_make_folds():
    _section("1. make_folds()")

    topics = _fake_topics(32)
    folds  = make_folds(topics, n_folds=5)
    _assert(len(folds) == 5, "5 folds produced")

    total_val_ids = []
    for i, (train_fold, val_fold) in enumerate(folds):
        _assert(len(train_fold) + len(val_fold) == 32,
                f"fold {i}: train+val = {len(train_fold)+len(val_fold)}")
        train_ids = {t["id"] for t in train_fold}
        val_ids   = {t["id"] for t in val_fold}
        _assert(train_ids.isdisjoint(val_ids), f"fold {i}: no overlap between train and val")
        total_val_ids.extend(val_ids)

    _assert(len(total_val_ids) == 32, "total val ids = 32 across all folds")
    _assert(len(set(total_val_ids)) == 32, "each topic appears in val exactly once")

    # reproducible — same order every call
    folds2 = make_folds(topics, n_folds=5)
    for (_, v1), (_, v2) in zip(folds, folds2):
        _assert([t["id"] for t in v1] == [t["id"] for t in v2], "fold is reproducible")

    # edge: fewer topics than folds — must not crash
    small = _fake_topics(3)
    folds_small = make_folds(small, n_folds=5)
    _assert(len(folds_small) == 5, "edge case n_topics < n_folds: no crash")


# ---------------------------------------------------------------------------
# 2. run_cv — offline mock retriever
# ---------------------------------------------------------------------------

def test_run_cv_offline():
    _section("2. run_cv() — offline mock retriever")

    doc_ids = [f"doc{i:03d}" for i in range(50)]
    topics  = _fake_topics(20)
    qrels        = _fake_qrels(topics, doc_ids, n_rel=3)
    # graded: first 2 docs score=2, next 1 score=1
    qrels_graded = {t["id"]: {d: 2 for d in doc_ids[:2]} | {doc_ids[2]: 1}
                    for t in topics}

    factory = lambda: _MockRetriever(doc_ids)
    cv = run_cv(
        factory, topics, qrels, qrels_graded, doc_ids,
        query_field="topic", n_folds=5, verbose=False,
    )

    # primary metric: NDCG@100
    _assert("mean_ndcg" in cv and "std_ndcg" in cv, "result has mean_ndcg + std_ndcg")
    _assert("ndcg_per_fold" in cv and len(cv["ndcg_per_fold"]) == 5, "5 per-fold NDCG values")
    _assert(0.0 <= cv["mean_ndcg"] <= 1.0, f"NDCG in [0,1]: {cv['mean_ndcg']:.4f}")
    _assert(cv["mean_ndcg"] > 0.0, "mock puts relevant docs first -- NDCG must be > 0")

    # secondary metrics still present
    _assert("mean_map" in cv and "std_map" in cv, "result has mean_map + std_map")
    _assert("map_per_fold" in cv and len(cv["map_per_fold"]) == 5, "5 per-fold MAP values")
    _assert("mean_mrr" in cv and "mean_p10" in cv, "result has mean_mrr + mean_p10")
    _assert(0.0 <= cv["mean_map"] <= 1.0, f"MAP in [0,1]: {cv['mean_map']:.4f}")
    _assert(cv["mean_map"] > 0.0, "mock always puts relevant docs first -- MAP must be > 0")

    fold_range = max(cv["ndcg_per_fold"]) - min(cv["ndcg_per_fold"])
    _assert(fold_range < 0.5, f"fold NDCG range < 0.5: {fold_range:.4f}")
    print(f"  [info] mean_ndcg={cv['mean_ndcg']:.4f}  mean_map={cv['mean_map']:.4f}")


# ---------------------------------------------------------------------------
# 3. SweepResult unit tests (offline)
# ---------------------------------------------------------------------------

def test_sweep_result_unit():
    _section("3. SweepResult — unit tests (offline)")
    import pandas as pd

    # ── 3a. lmdir ──────────────────────────────────────────────────────────
    r = _make_lmdir_result()
    _assert(r.kind == "lmdir", "kind == lmdir")
    _assert(r.param_col == "mu", "param_col == mu")

    df = r.to_dataframe()
    _assert(isinstance(df, pd.DataFrame), "to_dataframe() returns DataFrame")
    _assert(list(df.columns) == ["μ", "NDCG@100", "±std", "MAP", "MRR", "ΔNDCG"],
            f"lmdir columns correct: {list(df.columns)}")
    _assert(len(df) == 3, f"3 rows (got {len(df)})")
    # first row must be baseline-marked (ΔNDCG == 0.0000 for μ=2000 which is baseline)
    # baseline is last row (sorted by ndcg desc, baseline=2000 is worst)
    _assert(df["ΔNDCG"].iloc[0] == "+0.1100", f"best delta = +0.1100 (got {df['ΔNDCG'].iloc[0]})")
    _assert(df["ΔNDCG"].iloc[-1] == "baseline", f"baseline row labelled 'baseline'")

    best = r.best
    _assert(isinstance(best, dict), "best returns dict")
    _assert(best["mu"] == 75, f"best mu = 75 (got {best['mu']})")
    _assert(best["mean_ndcg"] == 0.62, f"best ndcg = 0.62")

    base = r.baseline()
    _assert(base is not None, "baseline() not None")
    _assert(base["mu"] == 2000, f"baseline mu == 2000 (got {base['mu']})")

    rep = repr(r)
    _assert("lmdir" in rep and "mu=75" in rep, f"repr contains kind and best param: {rep!r}")

    # ── 3b. lmjm ──────────────────────────────────────────────────────────
    r2 = _make_lmjm_result()
    df2 = r2.to_dataframe()
    _assert(list(df2.columns) == ["λ", "NDCG@100", "±std", "MAP", "MRR", "ΔNDCG"],
            f"lmjm columns correct: {list(df2.columns)}")
    _assert(r2.best["lambda"] == 0.7, "lmjm best lambda = 0.7")
    base2 = r2.baseline()
    _assert(base2 is not None and abs(base2["lambda"] - 0.7) < 1e-9, "lmjm baseline lambda = 0.7")
    # rows sorted ascending by lambda: [0.1, 0.7]; baseline is last row (iloc[-1])
    _assert(df2["ΔNDCG"].iloc[-1] == "baseline", "lmjm: baseline row (λ=0.7) labelled 'baseline'")

    # ── 3c. bm25 ──────────────────────────────────────────────────────────
    r3 = _make_bm25_result()
    df3 = r3.to_dataframe()
    _assert(list(df3.columns) == ["k1", "b", "NDCG@100", "±std", "MAP", "ΔNDCG"],
            f"bm25 columns correct: {list(df3.columns)}")
    _assert(len(df3) == 4, f"4 bm25 rows (got {len(df3)})")
    _assert(r3.best["k1"] == 1.5, f"bm25 best k1=1.5 (got {r3.best['k1']})")
    base3 = r3.baseline()
    _assert(base3 is not None and base3["k1"] == 1.2 and base3["b"] == 0.75,
            "bm25 baseline = (k1=1.2, b=0.75)")

    # ── 3d. encoder ────────────────────────────────────────────────────────
    r4 = _make_encoder_result()
    df4 = r4.to_dataframe()
    _assert(list(df4.columns) == ["Encoder", "NDCG@100", "MAP", "MRR", "P@10", "ΔNDCG"],
            f"encoder columns correct: {list(df4.columns)}")
    _assert(r4.best["alias"] == "medcpt", f"encoder best = medcpt (got {r4.best['alias']})")
    base4 = r4.baseline()
    _assert(base4 is not None and base4["alias"] == "msmarco", "encoder baseline = msmarco")

    # ── 3e. baseline() when baseline_id not in rows ────────────────────────
    r5 = SweepResult(
        rows=[{"mu": 100, "mean_ndcg": 0.5, "std_ndcg": 0.0, "mean_map": 0.4,
               "std_map": 0.0, "mean_mrr": 0.6, "mean_p10": 0.5}],
        param_col="mu", kind="lmdir", baseline_id=9999,
    )
    _assert(r5.baseline() is None, "baseline() returns None when id not in rows")


# ---------------------------------------------------------------------------
# 4. Plot smoke tests (offline, matplotlib backend = Agg)
# ---------------------------------------------------------------------------

def test_plot_smoke():
    _section("4. Plot smoke tests (offline, Agg backend)")
    import matplotlib
    matplotlib.use("Agg")   # no display needed
    import matplotlib.pyplot as plt
    from src.tuning.tuning_plots import (
        plot_lmdir_sweep,
        plot_lmjm_sweep,
        plot_bm25_sweep,
        plot_encoder_sweep,
        plot_rrf_sweep,
    )

    # plot_lmdir_sweep
    r_lmdir = _make_lmdir_result()
    fig = plot_lmdir_sweep(r_lmdir)
    _assert(isinstance(fig, plt.Figure), "plot_lmdir_sweep returns Figure")
    plt.close(fig)

    # plot_lmjm_sweep
    r_lmjm = _make_lmjm_result()
    fig = plot_lmjm_sweep(r_lmjm)
    _assert(isinstance(fig, plt.Figure), "plot_lmjm_sweep returns Figure")
    plt.close(fig)

    # plot_bm25_sweep
    r_bm25 = _make_bm25_result()
    fig = plot_bm25_sweep(r_bm25)
    _assert(isinstance(fig, plt.Figure), "plot_bm25_sweep returns Figure")
    plt.close(fig)

    # plot_encoder_sweep
    r_enc = _make_encoder_result()
    fig = plot_encoder_sweep(r_enc)
    _assert(isinstance(fig, plt.Figure), "plot_encoder_sweep returns Figure")
    plt.close(fig)

    # plot_rrf_sweep — with and without solo_scores
    _rrf_rows = [
        {"label": f"P{i}  k=60", "pair": f"P{i}", "rrf_k": 60,
         "mean_ndcg": 0.80 - i * 0.02, "std_ndcg": 0.01,
         "mean_map":  0.60 - i * 0.01, "std_map":  0.01,
         "mean_mrr":  0.85 - i * 0.01, "mean_p10": 0.70 - i * 0.01}
        for i in range(4)
    ]
    _solo = {
        "BM25":   {"ndcg": 0.77, "map": 0.55, "mrr": 0.83, "p10": 0.68},
        "LM-JM":  {"ndcg": 0.75, "map": 0.53, "mrr": 0.81, "p10": 0.66},
    }
    r_rrf_with = SweepResult(rows=_rrf_rows, param_col="label", kind="rrf",
                             baseline_id="P0", meta={"solo_scores": _solo})
    fig = plot_rrf_sweep(r_rrf_with)
    _assert(isinstance(fig, plt.Figure), "plot_rrf_sweep (with solo_scores) returns Figure")
    plt.close(fig)

    r_rrf_bare = SweepResult(rows=_rrf_rows, param_col="label", kind="rrf",
                             baseline_id="P0", meta={})
    fig = plot_rrf_sweep(r_rrf_bare)
    _assert(isinstance(fig, plt.Figure), "plot_rrf_sweep (no solo_scores) returns Figure")
    plt.close(fig)

    # save_path=None → no file created, no crash
    import tempfile, os as _os
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "lmdir.png"
        fig = plot_lmdir_sweep(r_lmdir, save_path=p)
        _assert(p.exists(), f"save_path PNG created: {p}")
        plt.close(fig)


# ---------------------------------------------------------------------------
# 5. run_bm25_sweep — integration (real index, real retriever)
# ---------------------------------------------------------------------------

def test_bm25_sweep(client, test_index: str, train_topics, qrels, qrels_graded, all_doc_ids):
    _section("5. run_bm25_sweep() — real index, 2 configs x 2 folds")

    result = run_bm25_sweep(
        client, test_index,
        train_topics = train_topics,
        qrels        = qrels,
        qrels_graded = qrels_graded,
        all_doc_ids  = all_doc_ids,
        k1_b_grid    = TEST_BM25_GRID,
        n_folds      = 2,
        query_field  = "topic",
    )

    _assert(isinstance(result, SweepResult), "run_bm25_sweep returns SweepResult")
    _assert(result.kind == "bm25", "kind == bm25")
    _assert(result.baseline_id == (1.2, 0.75), "baseline_id == (1.2, 0.75)")
    _assert(len(result.rows) == 2, f"2 rows (got {len(result.rows)})")
    _assert(result.rows == sorted(result.rows, key=lambda r: r["mean_ndcg"], reverse=True),
            "rows sorted by mean_ndcg descending")
    for r in result.rows:
        _assert("k1" in r and "b" in r, f"row has k1 and b: {r}")
        _assert(0.0 <= r["mean_ndcg"] <= 1.0, f"NDCG@100 in [0,1]: {r['mean_ndcg']:.4f}")

    import pandas as pd
    df = result.to_dataframe()
    _assert(isinstance(df, pd.DataFrame), "to_dataframe() returns DataFrame")
    _assert("NDCG@100" in df.columns, f"DataFrame has NDCG@100 column")
    print(f"  [info] {[(r['k1'], r['b'], round(r['mean_ndcg'],4)) for r in result.rows]}")


# ---------------------------------------------------------------------------
# 6. run_lmjm_sweep — integration
# ---------------------------------------------------------------------------

def test_lmjm_sweep(client, test_index: str, train_topics, qrels, qrels_graded, all_doc_ids):
    _section("6. run_lmjm_sweep() — real index, 2 lambdas x 2 folds")

    result = run_lmjm_sweep(
        client, test_index,
        train_topics = train_topics,
        qrels        = qrels,
        qrels_graded = qrels_graded,
        all_doc_ids  = all_doc_ids,
        lambdas      = TEST_LAMBDAS,
        n_folds      = 2,
        query_field  = "topic",
    )

    _assert(isinstance(result, SweepResult), "run_lmjm_sweep returns SweepResult")
    _assert(result.kind == "lmjm", "kind == lmjm")
    _assert(abs(result.baseline_id - 0.7) < 1e-9, "baseline_id == 0.7")
    _assert(len(result.rows) == 2, f"2 rows (got {len(result.rows)})")
    _assert(result.rows == sorted(result.rows, key=lambda r: r["mean_ndcg"], reverse=True),
            "rows sorted by mean_ndcg descending")
    for r in result.rows:
        _assert("lambda" in r, f"row has lambda key: {r}")
        _assert(0.0 <= r["mean_ndcg"] <= 1.0, f"NDCG@100 in [0,1]: {r['mean_ndcg']:.4f}")
    print(f"  [info] {[(r['lambda'], round(r['mean_ndcg'],4)) for r in result.rows]}")


# ---------------------------------------------------------------------------
# 7. run_lmdir_sweep — integration
# ---------------------------------------------------------------------------

def test_lmdir_sweep(client, test_index: str, train_topics, qrels, qrels_graded, all_doc_ids):
    _section("7. run_lmdir_sweep() — real index, 2 mu values x 2 folds")

    result = run_lmdir_sweep(
        client, test_index,
        train_topics = train_topics,
        qrels        = qrels,
        qrels_graded = qrels_graded,
        all_doc_ids  = all_doc_ids,
        mus          = TEST_MUS,
        n_folds      = 2,
        query_field  = "topic",
    )

    _assert(isinstance(result, SweepResult), "run_lmdir_sweep returns SweepResult")
    _assert(result.kind == "lmdir", "kind == lmdir")
    _assert(result.baseline_id == 2000, "baseline_id == 2000")
    _assert(len(result.rows) == 2, f"2 rows (got {len(result.rows)})")
    _assert(result.rows == sorted(result.rows, key=lambda r: r["mean_ndcg"], reverse=True),
            "rows sorted by mean_ndcg descending")
    for r in result.rows:
        _assert("mu" in r, f"row has mu key: {r}")
        _assert(0.0 <= r["mean_ndcg"] <= 1.0, f"NDCG@100 in [0,1]: {r['mean_ndcg']:.4f}")
    print(f"  [info] {[(r['mu'], round(r['mean_ndcg'],4)) for r in result.rows]}")


# ---------------------------------------------------------------------------
# 8. run_encoder_sweep — offline, pre-built random L2-normalised embeddings
# ---------------------------------------------------------------------------

def test_encoder_sweep_offline():
    _section("8. run_encoder_sweep() — offline, pre-built random embeddings")

    n_docs, n_queries, dim = 20, 6, 32
    topics  = _fake_topics(n_queries)
    doc_ids = [f"doc{i:03d}" for i in range(n_docs)]
    qrels        = _fake_qrels(topics, doc_ids, n_rel=3)
    # graded: first 2 docs = 2 (supporting), next 1 = 1 (neutral)
    qrels_graded = {t["id"]: {doc_ids[0]: 2, doc_ids[1]: 2, doc_ids[2]: 1}
                    for t in topics}

    rng = np.random.default_rng(42)

    def _rand_norm(n: int) -> np.ndarray:
        e = rng.standard_normal((n, dim)).astype(np.float32)
        return e / np.linalg.norm(e, axis=1, keepdims=True)

    enc_a_docs, enc_a_qrys = _rand_norm(n_docs), _rand_norm(n_queries)
    enc_b_docs, enc_b_qrys = _rand_norm(n_docs), _rand_norm(n_queries)

    # ── Test the internal helper directly (run_encoder_sweep needs a live model) ──
    metrics_a = _eval_encoder_exact_cosine(
        enc_a_docs, enc_a_qrys, topics, qrels, qrels_graded, doc_ids,
    )
    metrics_b = _eval_encoder_exact_cosine(
        enc_b_docs, enc_b_qrys, topics, qrels, qrels_graded, doc_ids,
    )

    _assert("NDCG@100" in metrics_a and "MAP" in metrics_a and "MRR" in metrics_a and "P@10" in metrics_a,
            "_eval_encoder_exact_cosine returns NDCG@100/MAP/MRR/P@10")
    _assert(len(metrics_a["per_query"]) == n_queries,
            f"per_query has {n_queries} entries")
    _assert(0.0 <= metrics_a["NDCG@100"] <= 1.0, f"NDCG@100 in [0,1]: {metrics_a['NDCG@100']:.4f}")
    _assert(0.0 <= metrics_a["MAP"] <= 1.0, f"MAP in [0,1]: {metrics_a['MAP']:.4f}")
    print(f"  [info] enc_a: NDCG@100={metrics_a['NDCG@100']:.4f}  MAP={metrics_a['MAP']:.4f}")
    print(f"  [info] enc_b: NDCG@100={metrics_b['NDCG@100']:.4f}  MAP={metrics_b['MAP']:.4f}")

    # ── Build a SweepResult manually from those metrics and verify its API ──
    import pandas as pd
    rows = sorted([
        {"alias": "enc_a", "ndcg": metrics_a["NDCG@100"], "map": metrics_a["MAP"],
         "mrr": metrics_a["MRR"], "p10": metrics_a["P@10"]},
        {"alias": "enc_b", "ndcg": metrics_b["NDCG@100"], "map": metrics_b["MAP"],
         "mrr": metrics_b["MRR"], "p10": metrics_b["P@10"]},
    ], key=lambda r: r["ndcg"], reverse=True)

    result = SweepResult(rows=rows, param_col="alias", kind="encoder",
                         baseline_id="msmarco", meta={})

    _assert(isinstance(result, SweepResult), "SweepResult constructed from offline metrics")
    _assert(result.kind == "encoder", "kind == encoder")
    _assert(result.baseline_id == "msmarco", "baseline_id == 'msmarco'")
    _assert(len(result.rows) == 2, f"2 rows (got {len(result.rows)})")
    _assert(result.rows == sorted(result.rows, key=lambda r: r["ndcg"], reverse=True),
            "rows sorted by ndcg descending")
    for r in result.rows:
        _assert("alias" in r, f"row has alias key: {r}")
        _assert(0.0 <= r["ndcg"] <= 1.0, f"ndcg in [0,1]: {r['ndcg']:.4f}")
        _assert(0.0 <= r["map"]  <= 1.0, f"map in [0,1]: {r['map']:.4f}")

    # baseline() returns None because "msmarco" is not in enc_a/enc_b aliases
    _assert(result.baseline() is None, "baseline() is None when 'msmarco' not in rows")

    df = result.to_dataframe()
    _assert(isinstance(df, pd.DataFrame), "to_dataframe() returns DataFrame")
    _assert("NDCG@100" in df.columns, "encoder DataFrame has NDCG@100 column")
    print(f"  [info] {[(r['alias'], round(r['ndcg'],4)) for r in result.rows]}")


# ---------------------------------------------------------------------------
# 9. field_ablation — integration (needs live index with BM25 default field)
# ---------------------------------------------------------------------------

def test_field_ablation(client, test_index: str, train_topics, qrels, qrels_graded, all_doc_ids):
    _section("9. field_ablation() — real index, BM25 over 3 query fields")

    winner = field_ablation(
        client       = client,
        index_name   = test_index,
        all_doc_ids  = all_doc_ids,
        train_topics = train_topics,
        qrels        = qrels,
        qrels_graded = qrels_graded,
    )

    _assert(winner in ("topic", "question", "concatenated"),
            f"winner is one of the 3 valid fields (got '{winner}')")
    print(f"  [info] winning field: '{winner}'")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("  src/tuning/__sweeper_test.py")
    print("=" * 60)

    # ── Offline tests (always run) ──────────────────────────────────────────
    test_make_folds()
    test_run_cv_offline()
    test_sweep_result_unit()
    test_plot_smoke()
    test_encoder_sweep_offline()

    # ── Integration tests (need live OpenSearch + a dedicated test index) ───
    _section("Integration test setup")

    client, test_index = _connect_opensearch()
    if client is None:
        print("\n  Integration tests SKIPPED (no OpenSearch connection).")
        print("\n" + "=" * 60)
        print("  Offline tests passed [ok]")
        print("=" * 60)
        return

    # Build mini topics + qrels against the MINI_CORPUS doc IDs.
    # Use 10 topics so we can do 2-fold CV with >=1 topic per fold.
    mini_topics = _fake_topics(10)
    mini_qrels  = _fake_qrels(mini_topics, MINI_DOC_IDS, n_rel=2)
    # graded: first doc = 2, second = 1 -- just need non-zero graded scores
    mini_qrels_graded = {t["id"]: {MINI_DOC_IDS[0]: 2, MINI_DOC_IDS[1]: 1}
                         for t in mini_topics}

    # Create the test index with all fields the sweeps need
    _build_test_index(client, test_index)

    try:
        test_bm25_sweep(client, test_index, mini_topics, mini_qrels, mini_qrels_graded, MINI_DOC_IDS)
        test_lmjm_sweep(client, test_index, mini_topics, mini_qrels, mini_qrels_graded, MINI_DOC_IDS)
        test_lmdir_sweep(client, test_index, mini_topics, mini_qrels, mini_qrels_graded, MINI_DOC_IDS)
        test_field_ablation(client, test_index, mini_topics, mini_qrels, mini_qrels_graded, MINI_DOC_IDS)
    finally:
        # always clean up -- never leave a test index behind
        _section("Cleanup")
        _delete_test_index(client, test_index)

    print("\n" + "=" * 60)
    print("  All tests passed [ok]")
    print("=" * 60)


if __name__ == "__main__":
    main()
