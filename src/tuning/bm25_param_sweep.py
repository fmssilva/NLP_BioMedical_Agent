"""
src/tuning/bm25_param_sweep.py

Sweep BM25 k1 and b parameters on the train set using 5-fold CV.

Current baseline: k1=1.2, b=0.75 (OpenSearch/Lucene defaults, via the 'BM25'
similarity on the 'contents' field in the index).

This script:
  1. Adds new BM25 similarity configurations and text fields to the existing index
  2. Bulk-updates the index to populate those new fields
  3. Runs 5-fold CV on the train set for each (k1, b) combination
  4. Prints a heatmap-style table and saves results/phase1/tuning/bm25_param_sweep.csv

Usage:
    python -m src.tuning.bm25_param_sweep           # run sweep, save CSV
    python -m src.tuning.bm25_param_sweep --show    # load and print existing CSV

Notes:
  - Adding settings requires closing + reopening the index (brief downtime <5s).
  - Grid size: 5 k1 × 4 b = 20 configurations. With 5-fold CV ≈ 100 retrieval runs.
  - Estimated time: ~5 minutes on a shared OpenSearch instance.
  - After this script, the index will have up to 20 extra text fields.
    To save index space, only add fields for promising regions after an initial
    coarse grid (uncomment COARSE_GRID below for a faster first pass).
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv()

from src.data.loader import load_corpus
from src.indexing.opensearch_client import get_client
from src.tuning.cv_utils import run_cv

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parameter grids
# ---------------------------------------------------------------------------

# Full grid — k1=1.8/2.0 removed because those BM25 similarity fields
# were never created in the index (MAP~0.01 = retrieval failure, not real).
# The k1=1.0->1.2->1.5 trend shows MAP has plateaued; see §12 in notebook.
K1_VALUES = [0.5, 0.8, 1.0, 1.2, 1.5]   # 1.2 = current default
B_VALUES  = [0.25, 0.5, 0.75, 1.0]       # 0.75 = current default

# Coarse grid for a quick first pass (6 configs, ~1.5 min)
# K1_VALUES = [0.8, 1.2, 1.5]
# B_VALUES  = [0.5, 0.75]

OUTPUT_DIR = ROOT / "results" / "phase1" / "tuning"
OUTPUT_CSV = OUTPUT_DIR / "bm25_param_sweep.csv"
QUERY_FIELD = "concatenated"
N_FOLDS = 5

BASELINE_K1 = 1.2
BASELINE_B  = 0.75


# ---------------------------------------------------------------------------
# Field name helper
# ---------------------------------------------------------------------------

def _field_name(k1: float, b: float) -> str:
    """Canonical field name for a (k1, b) pair: contents_bm25_k{k1str}_b{bstr}."""
    k1_str = str(k1).replace(".", "")   # 1.2 -> "12", 0.5 -> "05"
    b_str  = str(b).replace(".", "")    # 0.75 -> "075"
    return f"contents_bm25_k{k1_str}_b{b_str}"


def _sim_name(k1: float, b: float) -> str:
    k1_str = str(k1).replace(".", "")
    b_str  = str(b).replace(".", "")
    return f"bm25_k{k1_str}_b{b_str}_similarity"


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------

def add_bm25_fields(client, index_name: str, k1_values: list, b_values: list) -> None:
    """Add new BM25 similarity configs and text fields to the existing index."""
    # Check existing fields
    mapping = client.indices.get_mapping(index=index_name)
    actual_key = list(mapping.keys())[0]
    existing_fields = set(mapping[actual_key]["mappings"].get("properties", {}).keys())

    new_pairs = [
        (k1, b) for k1 in k1_values for b in b_values
        if _field_name(k1, b) not in existing_fields
        # Skip the baseline — it already exists as 'contents' with BM25 similarity
    ]

    if not new_pairs:
        print("[bm25_sweep] All BM25 fields already exist — skipping index update.")
        return

    print(f"[bm25_sweep] Adding BM25 fields for {len(new_pairs)} (k1, b) pairs...")

    # Close index to update similarity settings
    print("[bm25_sweep] Closing index...")
    client.indices.close(index=index_name)
    time.sleep(1)

    # Add new BM25 similarity configs
    new_sims = {
        _sim_name(k1, b): {"type": "BM25", "k1": k1, "b": b}
        for k1, b in new_pairs
    }
    client.indices.put_settings(index=index_name, body={"similarity": new_sims})
    print(f"[bm25_sweep] Added {len(new_sims)} similarity configs.")

    # Reopen index
    client.indices.open(index=index_name)
    time.sleep(2)
    print("[bm25_sweep] Index reopened.")

    # Add new text fields
    new_props = {
        _field_name(k1, b): {
            "type": "text",
            "analyzer": "standard",
            "similarity": _sim_name(k1, b),
        }
        for k1, b in new_pairs
    }
    client.indices.put_mapping(index=index_name, body={"properties": new_props})
    print(f"[bm25_sweep] Added {len(new_props)} text fields.")


def populate_bm25_fields(client, index_name: str, corpus: list[dict],
                          k1_values: list, b_values: list) -> None:
    """Populate new BM25 fields by copying 'contents' text into each new field."""
    new_pairs = [
        (k1, b) for k1 in k1_values for b in b_values
    ]
    fields_to_populate = [_field_name(k1, b) for k1, b in new_pairs]

    # Check if already populated
    if fields_to_populate:
        sample = client.search(
            index=index_name,
            body={"size": 1, "_source": [fields_to_populate[0]]},
        )
        hits = sample["hits"]["hits"]
        if hits and hits[0]["_source"].get(fields_to_populate[0]):
            print("[bm25_sweep] BM25 fields already populated — skipping bulk update.")
            return

    print(f"[bm25_sweep] Populating {len(fields_to_populate)} BM25 fields for {len(corpus)} docs...")

    assignments = " ".join([
        f"ctx._source.{f} = ctx._source.contents;"
        for f in fields_to_populate
    ])
    script = {"source": assignments, "lang": "painless"}

    try:
        response = client.update_by_query(
            index=index_name,
            body={"script": script, "query": {"match_all": {}}},
            wait_for_completion=True,
            refresh=True,
        )
        print(f"[bm25_sweep] update_by_query: updated {response.get('updated', 0)} docs [ok]")
    except Exception as e:
        logger.warning("update_by_query failed (%s) — falling back to individual updates.", e)
        _populate_individually(client, index_name, corpus, fields_to_populate)


def _populate_individually(client, index_name: str, corpus: list[dict],
                            field_names: list[str]) -> None:
    from opensearchpy.helpers import bulk

    def _actions():
        for doc in corpus:
            yield {
                "_op_type": "update",
                "_index": index_name,
                "_id": doc["id"],
                "doc": {f: doc["contents"] for f in field_names},
            }

    success, errors = bulk(client, _actions(), chunk_size=100, raise_on_error=False)
    client.indices.refresh(index=index_name)
    print(f"[bm25_sweep] Individual update: {success} succeeded, {len(errors)} errors")


# ---------------------------------------------------------------------------
# Retriever factory for each (k1, b) pair
# ---------------------------------------------------------------------------

def make_bm25_retriever_factory(client, index_name: str, k1: float, b: float):
    """
    Return a factory for a BM25 retriever querying the field for (k1, b).

    The baseline (k1=1.2, b=0.75) uses the original 'contents' field.
    Other configs use 'contents_bm25_k{k1str}_b{bstr}'.
    """
    if k1 == BASELINE_K1 and b == BASELINE_B:
        field = "contents"  # baseline uses the original BM25 field
    else:
        field = _field_name(k1, b)

    class _BM25Retriever:
        def __init__(self):
            self._client = client
            self._index = index_name
            self._field = field

        def search(self, query: str, size: int = 100) -> list[tuple[str, float]]:
            body = {
                "size": size,
                "_source": ["doc_id"],
                "query": {"match": {self._field: {"query": query}}},
            }
            resp = self._client.search(body=body, index=self._index)
            return [(h["_source"]["doc_id"], h["_score"]) for h in resp["hits"]["hits"]]

    return lambda: _BM25Retriever()


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_bm25_sweep(
    client,
    index_name: str,
    train_topics: list[dict],
    qrels: dict,
    all_doc_ids: list[str],
    corpus: list[dict],
    k1_values: list = K1_VALUES,
    b_values: list = B_VALUES,
    n_folds: int = N_FOLDS,
) -> list[dict]:
    """
    Run BM25 (k1, b) grid sweep on the train set.
    Returns a list of result dicts sorted by mean_map descending.
    """
    add_bm25_fields(client, index_name, k1_values, b_values)
    populate_bm25_fields(client, index_name, corpus, k1_values, b_values)

    results = []
    n_configs = len(k1_values) * len(b_values)
    print(f"\nRunning BM25 (k1, b) sweep - {n_configs} configs x {n_folds} folds...")

    # Print header for heatmap table
    header_label = "k1 / b"
    print(f"\n{header_label:>8}", end="")
    for b in b_values:
        print(f"  b={b:<6}", end="")
    print()
    print("-" * (10 + 10 * len(b_values)))

    # Collect all results first, then print heatmap
    table = {k1: {} for k1 in k1_values}

    config_num = 0
    for k1 in k1_values:
        for b in b_values:
            config_num += 1
            baseline_marker = " (baseline)" if k1 == BASELINE_K1 and b == BASELINE_B else ""
            factory = make_bm25_retriever_factory(client, index_name, k1, b)
            cv = run_cv(
                factory, train_topics, qrels, all_doc_ids,
                query_field=QUERY_FIELD, n_folds=n_folds, verbose=False,
            )
            table[k1][b] = cv["mean_map"]
            print(f"  [{config_num:2d}/{n_configs}] k1={k1}, b={b}: "
                  f"MAP={cv['mean_map']:.4f} +/- {cv['std_map']:.4f}{baseline_marker}")

            results.append({
                "k1":       k1,
                "b":        b,
                "mean_map": cv["mean_map"],
                "std_map":  cv["std_map"],
                "mean_mrr": cv["mean_mrr"],
                "mean_p10": cv["mean_p10"],
                "map_fold_1": cv["map_per_fold"][0] if len(cv["map_per_fold"]) > 0 else None,
                "map_fold_2": cv["map_per_fold"][1] if len(cv["map_per_fold"]) > 1 else None,
                "map_fold_3": cv["map_per_fold"][2] if len(cv["map_per_fold"]) > 2 else None,
                "map_fold_4": cv["map_per_fold"][3] if len(cv["map_per_fold"]) > 3 else None,
                "map_fold_5": cv["map_per_fold"][4] if len(cv["map_per_fold"]) > 4 else None,
            })

    # Print heatmap
    print(f"\nMAP heatmap (k1 rows x b cols):")
    hdr = "k1 / b"
    print(f"{hdr:>8}", end="")
    for b in b_values:
        print(f"  b={b:<6}", end="")
    print()
    print("-" * (10 + 10 * len(b_values)))
    for k1 in k1_values:
        k1_marker = " <-" if k1 == BASELINE_K1 else "   "
        print(f"k1={k1:<4}{k1_marker}", end="")
        for b in b_values:
            v = table[k1][b]
            b_marker = "*" if b == BASELINE_B else " "
            print(f"  {v:.4f}{b_marker}", end="")
        print()

    results_sorted = sorted(results, key=lambda r: r["mean_map"], reverse=True)
    best = results_sorted[0]
    baseline = next(r for r in results if r["k1"] == BASELINE_K1 and r["b"] == BASELINE_B)

    print(f"\n-> Best config: k1={best['k1']}, b={best['b']}  "
          f"(mean MAP={best['mean_map']:.4f} +/- {best['std_map']:.4f})")
    print(f"  Baseline k1={BASELINE_K1}, b={BASELINE_B}  MAP={baseline['mean_map']:.4f}")
    print(f"  Improvement: d_MAP = {best['mean_map'] - baseline['mean_map']:+.4f}")

    return results


def save_results(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved: {path}")


def load_and_print_results(path: Path) -> None:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    rows_sorted = sorted(rows, key=lambda r: float(r["mean_map"]), reverse=True)
    print(f"\nBM25 (k1, b) sweep results - top 10 (loaded from {path.name}):")
    print(f"{'k1':>5} | {'b':>5} | {'Mean MAP':>9} | {'std':>6} | {'Mean MRR':>9}")
    print("-" * 44)
    for row in rows_sorted[:10]:
        marker = " <- best" if row == rows_sorted[0] else ""
        print(
            f"{float(row['k1']):>5.2f} | {float(row['b']):>5.2f} | "
            f"{float(row['mean_map']):>9.4f} | {float(row['std_map']):>6.4f} | "
            f"{float(row['mean_mrr']):>9.4f}{marker}"
        )


# ---------------------------------------------------------------------------
# Convenience entry-point callable from notebooks or other scripts
# ---------------------------------------------------------------------------

def run_bm25_sweep_and_save(
    client=None,
    index_name: str = "",
    train_topics: list[dict] | None = None,
    qrels: dict | None = None,
    all_doc_ids: list[str] | None = None,
    corpus: list[dict] | None = None,
    output_csv: str | Path = OUTPUT_CSV,
    k1_values: list[float] = K1_VALUES,
    b_values: list[float] = B_VALUES,
    n_folds: int = N_FOLDS,
) -> list[dict]:
    """
    Run the BM25 k1/b parameter sweep and save results to CSV. Always overwrites existing CSV.

    Args:
        client:       OpenSearch client (required)
        index_name:   index to query (required)
        train_topics: list of train topic dicts (loaded from splits if None)
        qrels:        binary qrels dict (loaded from disk if None)
        all_doc_ids:  list of all corpus PMIDs (loaded from corpus if None)
        corpus:       list of corpus dicts, needed for index update (loaded if None)
        output_csv:   path to write CSV results
        k1_values:    k1 grid (default: K1_VALUES from module)
        b_values:     b grid (default: B_VALUES from module)
        n_folds:      number of CV folds

    Returns:
        list of result dicts sorted by mean_map descending
    """
    output_csv = Path(output_csv)

    if train_topics is None:
        with open(ROOT / "results" / "splits" / "train_queries.json") as f:
            train_topics = json.load(f)
    if qrels is None:
        with open(ROOT / "results" / "qrels.json") as f:
            qrels = json.load(f)
    if corpus is None:
        corpus = load_corpus(ROOT / "data" / "filtered_pubmed_abstracts.txt")
    if all_doc_ids is None:
        all_doc_ids = [doc["id"] for doc in corpus]
    if client is None:
        client = get_client()
    if not index_name:
        index_name = os.getenv("OPENSEARCH_INDEX", "")
    assert index_name, "OPENSEARCH_INDEX not set"

    n_configs = len(k1_values) * len(b_values)
    print("=" * 72)
    print("BM25 k1/b Parameter Sweep - 5-fold CV on train set")
    print(f"Grid: {len(k1_values)} k1 x {len(b_values)} b = {n_configs} configs")
    print(f"Index: {index_name}  |  Train topics: {len(train_topics)}  |  Folds: {n_folds}")
    print("=" * 72)

    results = run_bm25_sweep(
        client, index_name, train_topics, qrels, all_doc_ids, corpus,
        k1_values=k1_values, b_values=b_values, n_folds=n_folds,
    )
    save_results(results, output_csv)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Sweep BM25 k1/b on train set")
    parser.add_argument("--show",  action="store_true", help="Load and print existing CSV")
    parser.add_argument("--folds", type=int, default=N_FOLDS, help="Number of CV folds")
    parser.add_argument("--coarse", action="store_true", help="Use smaller 3×2 grid for quick test")
    args = parser.parse_args()

    if args.show:
        if OUTPUT_CSV.exists():
            load_and_print_results(OUTPUT_CSV)
        else:
            print(f"No results file found at {OUTPUT_CSV}. Run without --show first.")
        sys.exit(0)

    k1_vals = [0.8, 1.2, 1.5] if args.coarse else K1_VALUES
    b_vals  = [0.5, 0.75]    if args.coarse else B_VALUES

    run_bm25_sweep_and_save(k1_values=k1_vals, b_values=b_vals, n_folds=args.folds)
