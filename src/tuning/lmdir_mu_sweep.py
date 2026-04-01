"""
src/tuning/lmdir_mu_sweep.py

Sweep LM-Dirichlet μ values on the train set using 5-fold CV.

Current baseline: μ=2000 (index field: contents_lmdir)
Problem: μ=2000 is ~13× the mean document length (~150 words) — likely wrong.
Standard theory: μ ≈ mean document length gives the best smoothing prior.

This script:
  1. Adds new similarity configurations and text fields to the existing index
     for μ ∈ [100, 200, 500, 1000] (μ=2000 already exists as contents_lmdir)
  2. Bulk-updates the index to populate those new fields
  3. Runs 5-fold CV on the train set for each μ value
  4. Prints a comparison table and saves results/phase1/tuning/lmdir_mu_sweep.csv

Usage:
    python -m src.tuning.lmdir_mu_sweep           # run sweep, save CSV
    python -m src.tuning.lmdir_mu_sweep --show    # load and print existing CSV

Notes:
  - Adding settings requires closing + reopening the index (brief downtime <5s).
  - The index mapping must have dynamic="strict" relaxed to add new fields.
  - After this script, the index will have 4 extra text fields (contents_lmdir_100,
    contents_lmdir_200, contents_lmdir_500, contents_lmdir_1000).
  - All new fields hold the SAME text as contents_lmdir — only the similarity differs.
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

# ---------------------------------------------------------------------------
# Setup path so this runs as   python -m src.tuning.lmdir_mu_sweep
# ---------------------------------------------------------------------------
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
# Constants
# ---------------------------------------------------------------------------

MU_VALUES = [50, 75, 100, 200, 500, 1000, 2000]  # 2000 = existing baseline; 50/75 confirm no gain below 100
OUTPUT_DIR = ROOT / "results" / "phase1" / "tuning"
OUTPUT_CSV = OUTPUT_DIR / "lmdir_mu_sweep.csv"
QUERY_FIELD = "concatenated"
N_FOLDS = 5


# ---------------------------------------------------------------------------
# Index helpers — add new LM-Dir similarity fields
# ---------------------------------------------------------------------------

def add_lmdir_fields(client, index_name: str, mu_values: list[int]) -> None:
    """
    Add new LM-Dirichlet similarity configurations and text fields to an
    existing OpenSearch index.

    Requires closing the index to update similarity settings, then reopening.
    The dynamic mapping is temporarily relaxed to allow new fields.

    Only adds fields for μ values that don't already exist (idempotent).
    """
    # Check which fields already exist
    mapping = client.indices.get_mapping(index=index_name)
    actual_key = list(mapping.keys())[0]
    existing_fields = set(mapping[actual_key]["mappings"].get("properties", {}).keys())

    new_mus = [
        mu for mu in mu_values
        if mu != 2000  # μ=2000 already exists as contents_lmdir
        and f"contents_lmdir_{mu}" not in existing_fields
    ]

    if not new_mus:
        print("[lmdir_sweep] All LM-Dir fields already exist — skipping index update.")
        return

    print(f"[lmdir_sweep] Adding LM-Dir fields for μ ∈ {new_mus}...")

    # Step 1: Close index (required to update similarity settings)
    print("[lmdir_sweep] Closing index to update similarity settings...")
    client.indices.close(index=index_name)
    time.sleep(1)

    # Step 2: Add new similarity configurations
    new_sims = {
        f"lmdir_{mu}_similarity": {"type": "LMDirichlet", "mu": mu}
        for mu in new_mus
    }
    client.indices.put_settings(index=index_name, body={"similarity": new_sims})
    print(f"[lmdir_sweep] Added similarity configs: {list(new_sims.keys())}")

    # Step 3: Reopen index
    client.indices.open(index=index_name)
    time.sleep(2)  # wait for shard recovery
    print("[lmdir_sweep] Index reopened.")

    # Step 4: Add new text fields using those similarities
    # Note: dynamic=strict means we must explicitly add fields via put_mapping
    new_props = {
        f"contents_lmdir_{mu}": {
            "type": "text",
            "analyzer": "standard",
            "similarity": f"lmdir_{mu}_similarity",
        }
        for mu in new_mus
    }
    client.indices.put_mapping(index=index_name, body={"properties": new_props})
    print(f"[lmdir_sweep] Added text fields: {list(new_props.keys())}")


def populate_lmdir_fields(client, index_name: str, corpus: list[dict], mu_values: list[int]) -> None:
    """
    Populate the new LM-Dir fields by updating existing docs.

    Each doc already has 'contents_lmdir' — we copy the same text into the
    new fields. The similarity is applied at index time per field definition,
    not based on the text content.

    Uses update_by_query with a script that copies 'contents_lmdir' to each
    new field. Falls back to individual doc updates if update_by_query fails.
    """
    new_mus = [mu for mu in mu_values if mu != 2000]
    if not new_mus:
        return

    # Check if fields are already populated (sample the first doc)
    first_field = f"contents_lmdir_{new_mus[0]}"
    sample = client.search(
        index=index_name,
        body={"size": 1, "_source": [first_field]},
    )
    hits = sample["hits"]["hits"]
    if hits and hits[0]["_source"].get(first_field):
        print(f"[lmdir_sweep] Fields already populated — skipping bulk update.")
        return

    print(f"[lmdir_sweep] Populating {len(new_mus)} new LM-Dir fields for {len(corpus)} docs...")

    # Build script that copies contents_lmdir to each new field
    assignments = " ".join([
        f"ctx._source.contents_lmdir_{mu} = ctx._source.contents_lmdir;"
        for mu in new_mus
    ])
    script = {"source": assignments, "lang": "painless"}

    try:
        response = client.update_by_query(
            index=index_name,
            body={"script": script, "query": {"match_all": {}}},
            wait_for_completion=True,
            refresh=True,
        )
        updated = response.get("updated", 0)
        print(f"[lmdir_sweep] update_by_query: updated {updated} docs [ok]")
    except Exception as e:
        logger.warning("update_by_query failed (%s) — falling back to individual updates.", e)
        _populate_individually(client, index_name, corpus, new_mus)


def _populate_individually(client, index_name: str, corpus: list[dict], new_mus: list[int]) -> None:
    """Fallback: update each doc individually with new field values (slower)."""
    from opensearchpy.helpers import bulk

    def _actions():
        for doc in corpus:
            yield {
                "_op_type": "update",
                "_index": index_name,
                "_id": doc["id"],
                "doc": {f"contents_lmdir_{mu}": doc["contents"] for mu in new_mus},
            }

    success, errors = bulk(client, _actions(), chunk_size=100, raise_on_error=False)
    client.indices.refresh(index=index_name)
    print(f"[lmdir_sweep] Individual update: {success} succeeded, {len(errors)} errors")


# ---------------------------------------------------------------------------
# Retriever factory for each μ
# ---------------------------------------------------------------------------

def make_lmdir_retriever_factory(client, index_name: str, mu: int):
    """
    Return a factory callable that creates a LMDirichletRetriever for a given μ.

    For μ=2000: uses the existing 'contents_lmdir' field.
    For other μ: uses 'contents_lmdir_{mu}' field.
    """
    from src.retrieval.lm_dirichlet import LMDirichletRetriever

    field = "contents_lmdir" if mu == 2000 else f"contents_lmdir_{mu}"

    # We need a retriever that queries the right field
    # LMDirichletRetriever currently hardcodes the field name.
    # We create a simple wrapper that searches the correct field.
    class _LMDirRetriever:
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

    return lambda: _LMDirRetriever()


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_lmdir_sweep(
    client,
    index_name: str,
    train_topics: list[dict],
    qrels: dict,
    all_doc_ids: list[str],
    corpus: list[dict],
    n_folds: int = N_FOLDS,
) -> list[dict]:
    """
    Run LM-Dir μ sweep on the train set.

    Returns a list of result dicts, one per μ value, sorted by mean_map descending.
    """
    # Add new fields to index (idempotent)
    add_lmdir_fields(client, index_name, MU_VALUES)
    populate_lmdir_fields(client, index_name, corpus, MU_VALUES)

    results = []
    print(f"\nRunning LM-Dir μ sweep ({len(MU_VALUES)} values × {n_folds} folds)...")
    print(f"{'μ':>6} | {'Mean MAP':>9} | {'Std MAP':>8} | {'Mean MRR':>9} | {'MAP per fold'}")
    print("-" * 72)

    for mu in MU_VALUES:
        factory = make_lmdir_retriever_factory(client, index_name, mu)
        cv = run_cv(
            factory, train_topics, qrels, all_doc_ids,
            query_field=QUERY_FIELD, n_folds=n_folds, verbose=False,
        )

        baseline_marker = " <- baseline" if mu == 2000 else ""
        fold_maps = [f"{v:.4f}" for v in cv["map_per_fold"]]
        print(
            f"{mu:>6} | {cv['mean_map']:>9.4f} | {cv['std_map']:>8.4f} | "
            f"{cv['mean_mrr']:>9.4f} | {fold_maps}{baseline_marker}"
        )

        results.append({
            "mu":        mu,
            "mean_map":  cv["mean_map"],
            "std_map":   cv["std_map"],
            "mean_mrr":  cv["mean_mrr"],
            "mean_p10":  cv["mean_p10"],
            "map_fold_1": cv["map_per_fold"][0] if len(cv["map_per_fold"]) > 0 else None,
            "map_fold_2": cv["map_per_fold"][1] if len(cv["map_per_fold"]) > 1 else None,
            "map_fold_3": cv["map_per_fold"][2] if len(cv["map_per_fold"]) > 2 else None,
            "map_fold_4": cv["map_per_fold"][3] if len(cv["map_per_fold"]) > 3 else None,
            "map_fold_5": cv["map_per_fold"][4] if len(cv["map_per_fold"]) > 4 else None,
        })

    results_sorted = sorted(results, key=lambda r: r["mean_map"], reverse=True)
    best = results_sorted[0]
    print(f"\n>> Best mu: {best['mu']}  (mean MAP={best['mean_map']:.4f} +/- {best['std_map']:.4f})")
    baseline = next(r for r in results if r["mu"] == 2000)
    delta = best["mean_map"] - baseline["mean_map"]
    print(f"  Baseline μ=2000 mean MAP={baseline['mean_map']:.4f}")
    print(f"  Improvement: Δ MAP = {delta:+.4f}")

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
    print(f"\nLM-Dir μ sweep results (loaded from {path.name}):")
    print(f"{'μ':>6} | {'Mean MAP':>9} | {'±':>6} | {'Mean MRR':>9} | {'Mean P@10':>10}")
    print("-" * 52)
    for row in sorted(rows, key=lambda r: float(r["mean_map"]), reverse=True):
        marker = " <- best" if row == sorted(rows, key=lambda r: float(r["mean_map"]), reverse=True)[0] else ""
        print(
            f"{int(float(row['mu'])):>6} | {float(row['mean_map']):>9.4f} | "
            f"{float(row['std_map']):>6.4f} | {float(row['mean_mrr']):>9.4f} | "
            f"{float(row['mean_p10']):>10.4f}{marker}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Convenience entry-point callable from notebooks or other scripts
# ---------------------------------------------------------------------------

def run_lmdir_sweep_and_save(
    client=None,
    index_name: str = "",
    train_topics: list[dict] | None = None,
    qrels: dict | None = None,
    all_doc_ids: list[str] | None = None,
    corpus: list[dict] | None = None,
    output_csv: str | Path = OUTPUT_CSV,
    n_folds: int = N_FOLDS,
) -> list[dict]:
    """
    Run the LM-Dir μ sweep and save results to CSV. Always overwrites existing CSV.

    When called from a notebook, pass client / index_name / data objects that
    are already loaded. When called via __main__, they are loaded from defaults.

    Args:
        client:       OpenSearch client (required)
        index_name:   index to query (required)
        train_topics: list of train topic dicts (loaded from splits if None)
        qrels:        binary qrels dict (loaded from disk if None)
        all_doc_ids:  list of all corpus PMIDs (loaded from corpus if None)
        corpus:       list of corpus dicts, needed for index update (loaded if None)
        output_csv:   path to write CSV results
        n_folds:      number of CV folds

    Returns:
        list of result dicts sorted by mean_map descending
    """
    output_csv = Path(output_csv)

    # load defaults from disk when not supplied
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

    print("=" * 72)
    print("LM-Dirichlet μ Sweep — 5-fold CV on train set")
    print(f"Index: {index_name}  |  Train topics: {len(train_topics)}  |  Folds: {n_folds}")
    print("=" * 72)

    results = run_lmdir_sweep(
        client, index_name, train_topics, qrels, all_doc_ids, corpus, n_folds=n_folds
    )
    save_results(results, output_csv)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Sweep LM-Dirichlet μ on train set")
    parser.add_argument("--show",  action="store_true", help="Load and print existing CSV")
    parser.add_argument("--folds", type=int, default=N_FOLDS, help="Number of CV folds")
    args = parser.parse_args()

    if args.show:
        if OUTPUT_CSV.exists():
            load_and_print_results(OUTPUT_CSV)
        else:
            print(f"No results file found at {OUTPUT_CSV}. Run the sweep first.")
        sys.exit(0)

    run_lmdir_sweep_and_save(n_folds=args.folds)
