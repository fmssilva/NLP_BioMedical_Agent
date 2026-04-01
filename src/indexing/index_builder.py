"""
src/indexing/index_builder.py

OpenSearch index schema builder for the BioGen IR pipeline.

Public API:
    IndexSettings                          -- dataclass: HNSW + shard settings
    build_index_mapping(params, ...)       -- build (or extend) an index mapping
    create_index(client, index_name, ...)  -- build mapping + create index in one call
    verify_mapping(client, index_name)     -- print live settings/mappings summary
    get_live_fields(client, index_name)    -- return set of field names from live index

Design:
  - One shared index per project. Fields are ADDED to it as experiments grow.
  - All model params are passed as LISTS so callers can define any combination.
  - build_index_mapping() accepts an optional existing_mapping dict; new fields
    are merged in without touching existing ones (additive, never destructive).
  - create_index() is the single public entry-point: it calls build_index_mapping()
    internally so callers never need two steps.
  - OpenSearch bakes similarity params into fields at indexing time, so each
    (model, param) combination needs its own named field.

Field naming conventions:
  BM25   : contents_bm25_k{k1}_b{b}          e.g. contents_bm25_k15_b10
  LM-JM  : contents_lmjm_{lambda}            e.g. contents_lmjm_07
  LM-Dir : contents_lmdir_{mu}               e.g. contents_lmdir_75
  KNN    : embedding_{encoder_alias}         e.g. embedding_msmarco, embedding_medcpt

The baseline BM25 field is always named "contents" (OpenSearch default similarity).
The baseline KNN field is always named "embedding" (msmarco, initial index).
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from opensearchpy import OpenSearch

from src.indexing.opensearch_client import get_client, check_health


# ---------------------------------------------------------------------------
# Default best params (used when running python -m src.indexing.index_builder)
# Updated after Phase 1 CV sweep.  The notebook always passes its own values.
# ---------------------------------------------------------------------------

_BEST_PARAMS = {
    "bm25_k1_b_pairs": [(1.5, 1.0)],
    "lmjm_lambdas":    [0.7],
    "lmdir_mus":       [75],
    "encoders":        [("medcpt", "ncbi/MedCPT-Query-Encoder", 768)],
    #                    ↑ (alias, hf_model_id, dim)
}


# ---------------------------------------------------------------------------
# Index-level settings (hardware / HNSW topology)
# ---------------------------------------------------------------------------

@dataclass
class IndexSettings:
    """
    Index-level settings that apply globally (not per-field).

    KNN notes:
      ef_search     — candidates explored per query; higher = better recall, slower.
                      Applies index-wide (OpenSearch quirk: it's a setting, not per-field).
      ef_construct  — candidates when building the HNSW graph; higher = better graph, slower build.
      hnsw_m        — bi-directional links per node; higher = better connectivity, more RAM.
    """
    n_shards:         int = 4
    n_replicas:       int = 0
    refresh_interval: str = "-1"    # "-1" disables auto-refresh during bulk indexing (fast)
    ef_search:        int = 100     # should be >= k (number of results requested per query)
    ef_construct:     int = 256
    hnsw_m:           int = 48


# ---------------------------------------------------------------------------
# Internal field builders — one function per field family
# ---------------------------------------------------------------------------

def _float_tag(v: float) -> str:
    """0.7 → '07', 1.5 → '15', 2000 → '2000'"""
    return str(v).replace(".", "")


def _bm25_field(k1: float, b: float, analyzer: str) -> tuple[str, dict, dict]:
    """
    Returns (field_name, similarity_cfg, field_mapping).
    The baseline BM25 field (k1=1.2, b=0.75) uses the built-in "BM25" name.
    Custom k1/b pairs get a named similarity config.
    """
    if k1 == 1.2 and b == 0.75:
        # OpenSearch default — no custom similarity needed
        sim_name = "BM25"
        sim_cfg  = None
        fname    = "contents"
    else:
        sim_name = f"bm25_k{_float_tag(k1)}_b{_float_tag(b)}_similarity"
        sim_cfg  = {"type": "BM25", "k1": k1, "b": b}
        fname    = f"contents_bm25_k{_float_tag(k1)}_b{_float_tag(b)}"

    fmap = {"type": "text", "analyzer": analyzer, "similarity": sim_name}
    return fname, sim_cfg, fmap


def _lmjm_field(lam: float, analyzer: str) -> tuple[str, dict, dict]:
    sim_name = f"lmjm_{_float_tag(lam)}_similarity"
    sim_cfg  = {"type": "LMJelinekMercer", "lambda": lam}
    fname    = f"contents_lmjm_{_float_tag(lam)}"
    fmap     = {"type": "text", "analyzer": analyzer, "similarity": sim_name}
    return fname, sim_cfg, fmap


def _lmdir_field(mu: int, analyzer: str) -> tuple[str, dict, dict]:
    sim_name = f"lmdir_{mu}_similarity"
    sim_cfg  = {"type": "LMDirichlet", "mu": mu}
    fname    = f"contents_lmdir_{mu}"
    fmap     = {"type": "text", "analyzer": analyzer, "similarity": sim_name}
    return fname, sim_cfg, fmap


def _knn_field(alias: str, dim: int, settings: IndexSettings) -> tuple[str, dict]:
    """
    Returns (field_name, field_mapping).
    The baseline msmarco field is named "embedding"; others are "embedding_{alias}".
    """
    fname = "embedding" if alias == "msmarco" else f"embedding_{alias}"
    fmap  = {
        "type":      "knn_vector",
        "dimension": dim,
        "method": {
            "name":       "hnsw",
            "space_type": "innerproduct",   # == cosine when vectors are L2-normalised
            "engine":     "faiss",
            "parameters": {
                "ef_construction": settings.ef_construct,
                "m":               settings.hnsw_m,
            },
        },
    }
    return fname, fmap


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_index_mapping(
    bm25_k1_b_pairs: list[tuple[float, float]] | None = None,
    lmjm_lambdas:    list[float]                | None = None,
    lmdir_mus:       list[int]                  | None = None,
    encoders:        list[tuple[str, str, int]] | None = None,
    existing_mapping: dict | None = None,
    settings:        IndexSettings = None,
    analyzer:        str = "standard",
) -> dict:
    """
    Build (or extend) a full OpenSearch index body {settings, mappings}.

    All model params are lists — pass a list with one element for a single value.
    When ``existing_mapping`` is provided, new fields are merged in without
    touching existing ones (additive, safe to call repeatedly).

    Args:
        bm25_k1_b_pairs:  List of (k1, b) tuples.  e.g. [(1.2, 0.75), (1.5, 1.0)]
                          Default: [(1.2, 0.75)]  (OpenSearch default, baseline field "contents")
        lmjm_lambdas:     List of LM-JM λ values.  e.g. [0.1, 0.7]
                          Default: [0.7]
        lmdir_mus:        List of LM-Dir μ values.  e.g. [75, 2000]
                          Default: [75]
        encoders:         List of (alias, hf_model_id, dim) tuples.
                          e.g. [("msmarco", "sentence-transformers/msmarco-distilbert-base-v2", 768)]
                          alias "msmarco" → field "embedding"; others → "embedding_{alias}"
                          Default: [("msmarco", "...", 768)]
        existing_mapping: Existing {settings, mappings} dict from a previous call or
                          from client.indices.get(). New fields are merged in.
                          Pass None to build from scratch.
        settings:         IndexSettings dataclass (shards, replicas, HNSW params).
                          Defaults to IndexSettings() with sensible defaults.
        analyzer:         Text analyzer for all text fields.  Default "standard".

    Returns:
        dict with keys "settings" and "mappings" ready for client.indices.create().
    """
    if settings is None:
        settings = IndexSettings()

    # Apply defaults
    if bm25_k1_b_pairs is None:
        bm25_k1_b_pairs = [(1.2, 0.75)]
    if lmjm_lambdas is None:
        lmjm_lambdas = [0.7]
    if lmdir_mus is None:
        lmdir_mus = [75]
    if encoders is None:
        encoders = [("msmarco", "sentence-transformers/msmarco-distilbert-base-v2", 768)]

    # Collect new similarities and fields
    new_similarities: dict = {}
    new_fields: dict = {}

    # Always include the doc_id keyword field
    new_fields["doc_id"] = {"type": "keyword"}

    # BM25 fields
    for k1, b in bm25_k1_b_pairs:
        fname, sim_cfg, fmap = _bm25_field(k1, b, analyzer)
        if sim_cfg is not None:
            new_similarities[f"bm25_k{_float_tag(k1)}_b{_float_tag(b)}_similarity"] = sim_cfg
        new_fields[fname] = fmap

    # LM-JM fields
    for lam in lmjm_lambdas:
        fname, sim_cfg, fmap = _lmjm_field(lam, analyzer)
        new_similarities[f"lmjm_{_float_tag(lam)}_similarity"] = sim_cfg
        new_fields[fname] = fmap

    # LM-Dir fields
    for mu in lmdir_mus:
        fname, sim_cfg, fmap = _lmdir_field(mu, analyzer)
        new_similarities[f"lmdir_{mu}_similarity"] = sim_cfg
        new_fields[fname] = fmap

    # KNN embedding fields
    for alias, _hf_id, dim in encoders:
        fname, fmap = _knn_field(alias, dim, settings)
        new_fields[fname] = fmap

    # Merge with existing mapping (if given)
    if existing_mapping is not None:
        existing_sims = (
            existing_mapping.get("settings", {})
            .get("index", {})
            .get("similarity", {})
        )
        existing_props = (
            existing_mapping.get("mappings", {})
            .get("properties", {})
        )
        # Merge: existing wins on collision (never overwrite an existing field)
        merged_sims   = {**new_similarities, **existing_sims}
        merged_fields = {**new_fields, **existing_props}
    else:
        merged_sims   = new_similarities
        merged_fields = new_fields

    index_settings = {
        "index": {
            "number_of_shards":        settings.n_shards,
            "number_of_replicas":      settings.n_replicas,
            "refresh_interval":        settings.refresh_interval,
            "knn":                     "true",
            "knn.algo_param.ef_search": settings.ef_search,
        },
        "similarity": merged_sims,
    }

    mappings = {
        "dynamic":    "strict",
        "properties": merged_fields,
    }

    return {"settings": index_settings, "mappings": mappings}


def get_live_fields(client: OpenSearch, index_name: str) -> set[str]:
    """Return set of field names in the live index. Empty set if index doesn't exist."""
    if not client.indices.exists(index=index_name):
        return set()
    resp = client.indices.get_mapping(index=index_name)
    actual_key = list(resp.keys())[0]
    props = resp[actual_key]["mappings"].get("properties", {})
    return set(props.keys())


def get_live_field_types(client: OpenSearch, index_name: str) -> dict[str, str]:
    """
    Return {field_name: field_type} for all top-level fields in the live index.
    Returns empty dict if the index doesn't exist.

    Example:
        {"doc_id": "keyword", "contents": "text",
         "embedding": "knn_vector", "embedding_medcpt": "knn_vector"}
    """
    if not client.indices.exists(index=index_name):
        return {}
    resp = client.indices.get_mapping(index=index_name)
    actual_key = list(resp.keys())[0]
    props = resp[actual_key]["mappings"].get("properties", {})
    return {name: prop.get("type", "object") for name, prop in props.items()}


def delete_index(client: OpenSearch, index_name: str) -> None:
    """
    Delete the named OpenSearch index — all docs and mapping are lost.
    No-op (with a warning) if the index doesn't exist.
    Used for testing and full reset between experiments.
    """
    if not client.indices.exists(index=index_name):
        print(f"[index_builder] delete_index: '{index_name}' does not exist — nothing to delete.")
        return
    resp = client.indices.delete(index=index_name)
    if resp.get("acknowledged"):
        print(f"[index_builder] delete_index: '{index_name}' deleted [ok]")
    else:
        raise RuntimeError(f"Index deletion not acknowledged for '{index_name}'. Response: {resp}")


def create_index(
    client:          OpenSearch,
    index_name:      str,
    bm25_k1_b_pairs: list[tuple[float, float]] | None = None,
    lmjm_lambdas:    list[float]                | None = None,
    lmdir_mus:       list[int]                  | None = None,
    encoders:        list[tuple] | None = None,
    settings:        IndexSettings = None,
    analyzer:        str = "standard",
) -> dict:
    """
    Build the index mapping and create the index in OpenSearch in one call.

    Idempotent: if the index already exists, logs and returns without changes.

    Args: same as build_index_mapping() — see that docstring.
    encoders: accepts either:
      - list[(alias, hf_model_id, dim)]      -- explicit dim (legacy / manual)
      - list[(alias, hf_model_id, ndarray)]  -- from create_embeddings; dim inferred from array

    Returns:
        The mapping dict that was (or would have been) used for creation.
    """
    import numpy as np

    # normalise encoders: if third element is an ndarray, extract its dim
    if encoders is not None:
        normalised = []
        for entry in encoders:
            alias, hf_id, third = entry[0], entry[1], entry[2]
            dim = int(third.shape[1]) if isinstance(third, np.ndarray) else int(third)
            normalised.append((alias, hf_id, dim))
        encoders = normalised

    mapping = build_index_mapping(
        bm25_k1_b_pairs=bm25_k1_b_pairs,
        lmjm_lambdas=lmjm_lambdas,
        lmdir_mus=lmdir_mus,
        encoders=encoders,
        settings=settings,
        analyzer=analyzer,
    )

    if client.indices.exists(index=index_name):
        print(f"[index_builder] Index '{index_name}' already exists — skipping creation.")
        return mapping

    print(f"[index_builder] Creating index '{index_name}' ...")
    response = client.indices.create(index=index_name, body=mapping)
    if response.get("acknowledged"):
        fields = list(mapping["mappings"]["properties"].keys())
        print(f"[index_builder] Index '{index_name}' created [ok]  fields={fields}")
    else:
        raise RuntimeError(
            f"Index creation not acknowledged for '{index_name}'. Response: {response}"
        )

    return mapping






