import os
import sys
from dataclasses import dataclass
from pathlib import Path

from opensearchpy import OpenSearch

from src.indexing.opensearch_client import get_client, check_health


######################################################################
## Index schema builder — field definitions and index lifecycle.
## Field naming: contents_bm25_k{k1}_b{b}, contents_lmjm_{lam},
##               contents_lmdir_{mu}, embedding_{alias}.
######################################################################


# ---------------------------------------------------------------------------
# Default best params — for local CLI test only (python -m src.indexing.index_builder).
# SOURCE OF TRUTH for these values is the notebook constants cell (§1.2).
# Do NOT use these in src/ logic — accept params as arguments instead.
# Updated after Phase 1 CV sweep.
# ---------------------------------------------------------------------------

_BEST_PARAMS = {
    "bm25_k1_b_pairs": [(1.5, 1.0)],
    "lmjm_lambdas":    [0.7],
    "lmdir_mus":       [75],
    "encoders":        [("medcpt", "ncbi/MedCPT-Query-Encoder", 768)],
    #                    ^ (alias, hf_model_id, dim) -- alias -> embedding_{alias} field
}


# ---------------------------------------------------------------------------
# Index-level settings (hardware / HNSW topology)
# ---------------------------------------------------------------------------

@dataclass
class IndexSettings:
    """
    Index-level settings that apply globally to index (not per-field).

    KNN params:
      ef_search     — candidates explored per query; higher = better recall, slower.
                      Applies index-wide (OpenSearch quirk: it's a setting, not per-field).
      ef_construct  — candidates when building the HNSW graph; higher = better graph, slower build.
      hnsw_m        — bi-directional links per node; higher = better connectivity, more RAM.
    """
    n_shards:         int = 4
    n_replicas:       int = 0
    refresh_interval: str = "-1"    # "-1" disables auto-refresh during bulk indexing (fast)
    # knn params
    ef_search:        int = 100     # should be >= k (number of results requested per query)
    ef_construct:     int = 256
    hnsw_m:           int = 48


# ---------------------------------------------------------------------------
# Field builders — one function per field family
# ---------------------------------------------------------------------------

def float_tag(v: float) -> str:
    """Shared helper: 0.7 -> '07', 1.5 -> '15', 2000 -> '2000'. Used by indexing + retrieval."""
    return str(v).replace(".", "")


def _bm25_field(k1: float, b: float, analyzer: str) -> tuple[str, dict, dict]:
    # Example for BM25 with k1=1.5, b=1.0, we return:
    # ("contents_bm25_k15_b10", {"type": "BM25", "k1": 1.5, "b": 1.0}, {"type": "text", "analyzer": "standard", "similarity": "bm25_k15_b10_similarity"})  
    sim_name = f"bm25_k{float_tag(k1)}_b{float_tag(b)}_similarity"
    sim_cfg  = {"type": "BM25", "k1": k1, "b": b}
    fname    = f"contents_bm25_k{float_tag(k1)}_b{float_tag(b)}"
    fmap     = {"type": "text", "analyzer": analyzer, "similarity": sim_name}
    return fname, sim_cfg, fmap


def _lmjm_field(lam: float, analyzer: str) -> tuple[str, dict, dict]:
    # Example for LM-JM with lambda=0.7, we return:
    # ("contents_lmjm_07", {"type": "LMJelinekMercer", "lambda": 0.7}, {"type": "text", "analyzer": "standard", "similarity": "lmjm_07_similarity"})
    sim_name = f"lmjm_{float_tag(lam)}_similarity"
    sim_cfg  = {"type": "LMJelinekMercer", "lambda": lam}
    fname    = f"contents_lmjm_{float_tag(lam)}"
    fmap     = {"type": "text", "analyzer": analyzer, "similarity": sim_name}
    return fname, sim_cfg, fmap


def _lmdir_field(mu: int, analyzer: str) -> tuple[str, dict, dict]:
    # Example for LM-Dir with mu=75, we return:
    # ("contents_lmdir_75", {"type": "LMDirichlet", "mu": 75}, {"type": "text", "analyzer": "standard", "similarity": "lmdir_75_similarity"})
    sim_name = f"lmdir_{mu}_similarity"
    sim_cfg  = {"type": "LMDirichlet", "mu": mu}
    fname    = f"contents_lmdir_{mu}"
    fmap     = {"type": "text", "analyzer": analyzer, "similarity": sim_name}
    return fname, sim_cfg, fmap


def _knn_field(alias: str, dim: int, settings: IndexSettings) -> tuple[str, dict]:
    # Example for KNN with alias="msmarco", dim=768, we return:
    # ("embedding_msmarco", {"type": "knn_vector", "dimension": 768, "method": {"name": "hnsw", "space_type": "innerproduct", "engine": "faiss", "parameters": {"ef_construction": 256, "m": 48}}})
    fname = f"embedding_{alias}"
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
# Main Public Method
# ---------------------------------------------------------------------------

def build_index_mapping(
    bm25_k1_b_pairs:  list[tuple[float, float]]    = [],
    lmjm_lambdas:     list[float]                   = [],
    lmdir_mus:        list[int]                     = [],
    encoders:         list[tuple[str, str, int]]    = [],
    existing_mapping: dict                          | None = None,
    settings:         IndexSettings                 | None = None,
    analyzer:         str = "standard",
) -> dict:
    """
    Build (or extend) a full OpenSearch index body {settings, mappings}.
    Returns dict ready for client.indices.create().

    Pass only the field families you need — omit a list to skip that type entirely.

    encoders: (alias, hf_model_id, dim) — same constants as in the notebook,
              e.g. [("msmarco", "sentence-transformers/...", 768)].
    existing_mapping: merge new fields in without touching existing ones (additive).
    """
    if settings is None:
        settings = IndexSettings()

    # Collect new similarities and fields
    new_similarities: dict = {}
    new_fields: dict = {}

    # doc_id is always present
    new_fields["doc_id"] = {"type": "keyword"}

    # BM25 fields
    for k1, b in bm25_k1_b_pairs:
        fname, sim_cfg, fmap = _bm25_field(k1, b, analyzer)
        new_similarities[f"bm25_k{float_tag(k1)}_b{float_tag(b)}_similarity"] = sim_cfg
        new_fields[fname] = fmap

    # LM-JM fields
    for lam in lmjm_lambdas:
        fname, sim_cfg, fmap = _lmjm_field(lam, analyzer)
        new_similarities[f"lmjm_{float_tag(lam)}_similarity"] = sim_cfg
        new_fields[fname] = fmap

    # LM-Dir fields
    for mu in lmdir_mus:
        fname, sim_cfg, fmap = _lmdir_field(mu, analyzer)
        new_similarities[f"lmdir_{mu}_similarity"] = sim_cfg
        new_fields[fname] = fmap

    # KNN embedding fields — encoders: (alias, hf_model_id, dim)
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
    """
    if not client.indices.exists(index=index_name):
        print(f"[index_builder] delete_index: '{index_name}' does not exist — nothing to delete.")
        return
    resp = client.indices.delete(index=index_name)
    if resp.get("acknowledged"):
        print(f"[index_builder] delete_index: '{index_name}' deleted [ok]")
    else:
        raise RuntimeError(f"Index deletion not acknowledged for '{index_name}'. Response: {resp}")


def create_or_update_index(
    client:          OpenSearch,
    index_name:      str,
    bm25_k1_b_pairs: list[tuple[float, float]]  = [],
    lmjm_lambdas:    list[float]                 = [],
    lmdir_mus:       list[int]                   = [],
    encoders:        list[tuple[str, str, int]]  = [],
    settings:        IndexSettings               = None,
    analyzer:        str = "standard",
) -> None:
    """
    Single entry point for index lifecycle. Creates the index if new, or adds
    only the missing fields/similarities if it already exists. Fully idempotent.

    Branch A (new index):  builds full mapping + creates in one shot.
    Branch B (existing):   diffs live fields vs requested, adds only what's missing.
                           Similarity changes need close->put_settings->open (handled here).
                           KNN dim mismatches are warned but not auto-fixed — re-run
                           index_documents to overwrite the stored vectors.

    encoders: list of (alias, hf_model_id, dim) — same constants as in the notebook,
              e.g. [("msmarco", "sentence-transformers/...", 768)].
    """
    if settings is None:
        settings = IndexSettings()

    # ── Branch A: create from scratch ─────────────────────────────────────────
    if not client.indices.exists(index=index_name):
        mapping = build_index_mapping(
            bm25_k1_b_pairs = bm25_k1_b_pairs,
            lmjm_lambdas    = lmjm_lambdas,
            lmdir_mus       = lmdir_mus,
            encoders        = encoders,
            settings        = settings,
            analyzer        = analyzer,
        )
        print(f"[index_builder] Creating index '{index_name}' ...")
        response = client.indices.create(index=index_name, body=mapping)
        if response.get("acknowledged"):
            fields = list(mapping["mappings"]["properties"].keys())
            print(f"[index_builder] Index '{index_name}' created [ok]  ({len(fields)} fields)")
        else:
            raise RuntimeError(
                f"Index creation not acknowledged for '{index_name}'. Response: {response}"
            )
        return

    # ── Branch B: index exists — diff and add only what is missing ────────────
    print(f"[index_builder] Index '{index_name}' exists — checking for missing fields ...")
    live_fields = get_live_fields(client, index_name)
    live_sims_cfg = (
        client.indices.get_settings(index=index_name)
        .get(index_name, {}).get("settings", {}).get("index", {}).get("similarity", {})
    )
    live_sim_names = set(live_sims_cfg.keys())
    print(f"  Live fields: {len(live_fields)}  |  Live sim configs: {len(live_sim_names)}")

    new_sims  = {}  # sim configs not yet in the index settings
    new_props = {}  # field mappings not yet in the index

    for k1, b in bm25_k1_b_pairs:
        fname, sim_cfg, fmap = _bm25_field(k1, b, analyzer)
        if fname not in live_fields:
            new_props[fname] = fmap
            sim_name = fmap.get("similarity")
            if sim_name and sim_name not in live_sim_names:
                new_sims[sim_name] = sim_cfg
            print(f"  [+] BM25  : {fname}")
        else:
            print(f"  [=] BM25  : {fname}")

    for lam in lmjm_lambdas:
        fname, sim_cfg, fmap = _lmjm_field(lam, analyzer)
        if fname not in live_fields:
            new_props[fname] = fmap
            sim_name = fmap.get("similarity")
            if sim_name and sim_name not in live_sim_names:
                new_sims[sim_name] = sim_cfg
            print(f"  [+] LMJM  : {fname}")
        else:
            print(f"  [=] LMJM  : {fname}")

    for mu in lmdir_mus:
        fname, sim_cfg, fmap = _lmdir_field(mu, analyzer)
        if fname not in live_fields:
            new_props[fname] = fmap
            sim_name = fmap.get("similarity")
            if sim_name and sim_name not in live_sim_names:
                new_sims[sim_name] = sim_cfg
            print(f"  [+] LMDir : {fname}")
        else:
            print(f"  [=] LMDir : {fname}")

    # For KNN fields that already exist, check dim matches — can't change it after creation.
    # Warn only; caller must re-run index_documents to overwrite the stored vectors.
    live_props = (
        client.indices.get_mapping(index=index_name)
        .get(index_name, {}).get("mappings", {}).get("properties", {})
    )
    for alias, _model_name, dim in encoders:
        fname, fmap = _knn_field(alias, dim, settings)
        if fname not in live_fields:
            new_props[fname] = fmap
            print(f"  [+] KNN   : {fname}  (dim={dim})")
        else:
            live_dim = live_props.get(fname, {}).get("dimension")
            if live_dim is not None and int(live_dim) != dim:
                print(f"  [!] KNN dim mismatch '{fname}': live={live_dim}, requested={dim}")
            else:
                print(f"  [=] KNN   : {fname}  (dim={dim})")

    # Similarity changes require close -> put_settings -> open
    if new_sims:
        print(f"  Adding {len(new_sims)} sim config(s) — closing index ...")
        client.indices.close(index=index_name)
        try:
            client.indices.put_settings(index=index_name,
                                        body={"index": {"similarity": new_sims}})
            print("  put_settings [ok]")
        finally:
            client.indices.open(index=index_name)
            print("  Index re-opened.")

    if new_props:
        print(f"  put_mapping: {len(new_props)} new field(s): {sorted(new_props)}")
        client.indices.put_mapping(index=index_name, body={"properties": new_props})
        print("  put_mapping [ok]")
    else:
        print("  Index mapping is already complete — nothing to add.")


#################################################################
##                  LOCAL TEST                                 ##
#################################################################
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    client     = get_client()
    index_name = os.getenv("OPENSEARCH_INDEX", "biogen_test")

    check_health(client)
    create_or_update_index(
        client,
        index_name,
        bm25_k1_b_pairs=_BEST_PARAMS["bm25_k1_b_pairs"],
        lmjm_lambdas=_BEST_PARAMS["lmjm_lambdas"],
        lmdir_mus=_BEST_PARAMS["lmdir_mus"],
        encoders=_BEST_PARAMS["encoders"],
    )
    live_fields = get_live_fields(client, index_name)
    live_types  = get_live_field_types(client, index_name)
    print(f"\nLive fields ({len(live_fields)}): {sorted(live_fields)}")
    print(f"Field types: {live_types}")
    print("\n[index_builder] local test done.")

