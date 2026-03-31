"""
src/evaluation/metrics.py

Standard IR evaluation metrics — implemented from scratch following Lab03 exactly.
No ranx, no ir-measures, no external IR library.

Public API:
    precision_at_k(ranking, relevance, k)                    -> float
    recall_at_k(ranking, relevance, k)                        -> float
    average_precision(ranking, relevance)                     -> float
    mean_average_precision(queries)                           -> float   (TREC: excludes 0-rel topics)
    reciprocal_rank(ranking, relevance)                       -> float
    mean_reciprocal_rank(queries)                             -> float   (TREC: excludes 0-rel topics)
    ndcg_at_k(ranking, relevance_scores, k)                   -> float   (graded relevance)
    mean_ndcg_at_k(queries, k)                                -> float   (graded, excludes 0-rel topics)
    pr_curve(ranking, relevance)                              -> (recalls, precisions)
    interpolated_pr_curve(recalls, precisions, n=11)          -> (recall_levels, interp_precisions)
    mean_pr_curve(queries, n=11)                              -> (recall_levels, mean_precisions)
    results_to_ranking(results, qrels_set, all_doc_ids)       -> (relevance, ranking)
    results_to_ranking_graded(results, qrels_graded, all_doc_ids) -> (scores, ranking)

`queries` format for binary metrics: list of (relevance_labels, ranking)
  - relevance_labels: list[bool] parallel to all_doc_ids
  - ranking: list[int] — doc indices into all_doc_ids, ordered by retrieval rank

`queries` format for graded metrics: list of (relevance_scores, ranking)
  - relevance_scores: list[float] parallel to all_doc_ids (0.0 = not relevant)
  - ranking: list[int] — same integer-index format

Lab03 reference: Lab03_Retrieval_Evaluation.ipynb lines 86-488, 756-805.
"""

import logging
import warnings

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core single-query metrics (Lab03 lines 86-119, 268-305, 406-428, 180-244)
# ---------------------------------------------------------------------------

# Fraction of top-k retrieved documents that are relevant.
def precision_at_k(ranking: list[int], relevance: list[bool], k: int) -> float:
    """Lab03 pattern: count relevant in top-k, divide by k."""
    top_k = ranking[:k]
    n_relevant_in_top_k = sum(relevance[doc_id] for doc_id in top_k)
    return n_relevant_in_top_k / k


# Fraction of all relevant documents found in top-k.
def recall_at_k(ranking: list[int], relevance: list[bool], k: int) -> float:
    """Lab03 pattern: count relevant in top-k, divide by total relevant."""
    top_k = ranking[:k]
    n_relevant_in_top_k = sum(relevance[doc_id] for doc_id in top_k)
    total_relevant = sum(relevance)
    return n_relevant_in_top_k / total_relevant if total_relevant > 0 else 0.0


# Area under the precision-recall curve for a single query (Lab03 lines 268-305).
def average_precision(ranking: list[int], relevance: list[bool]) -> float:
    """
    Mean of P@k at each rank where a relevant document appears.
    Returns 0.0 if no relevant documents (avoids divide-by-zero).
    """
    total_relevant = sum(relevance)
    if total_relevant == 0:
        return 0.0

    score = 0.0
    n_relevant_seen = 0
    for rank, doc_id in enumerate(ranking, start=1):
        if relevance[doc_id]:
            n_relevant_seen += 1
            score += n_relevant_seen / rank  # precision at this rank

    return score / total_relevant


# 1/rank of the first relevant document (Lab03 lines 406-428).
def reciprocal_rank(ranking: list[int], relevance: list[bool]) -> float:
    """Returns 1/rank_of_first_relevant_doc, or 0.0 if none found."""
    for rank, doc_id in enumerate(ranking, start=1):
        if relevance[doc_id]:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Multi-query aggregates (TREC-standard: exclude 0-relevant-doc topics)
# ---------------------------------------------------------------------------

def _filter_zero_relevant(queries: list[tuple]) -> list[tuple]:
    """
    Remove queries with no relevant documents and log a warning per TREC standard.
    Lab03 uses np.mean() which includes them (AP=0); we filter for TREC correctness.
    """
    filtered = []
    n_excluded = 0
    for labels, ranking in queries:
        if sum(labels) == 0:
            n_excluded += 1
        else:
            filtered.append((labels, ranking))

    if n_excluded > 0:
        logger.warning(
            "[metrics] Excluded %d topic(s) with 0 relevant documents "
            "(TREC standard: do not penalise MAP/MRR for unjudged topics).",
            n_excluded,
        )

    return filtered


# MAP across queries — excludes 0-relevant topics (TREC standard).
def mean_average_precision(queries: list[tuple]) -> float:
    """
    Mean Average Precision over a list of (relevance_labels, ranking) pairs.

    TREC standard: topics with 0 relevant documents are EXCLUDED from the mean.
    Lab03 uses np.mean() which includes them — our implementation filters them first.
    """
    valid = _filter_zero_relevant(queries)
    if not valid:
        return 0.0
    return float(np.mean([average_precision(ranking, labels) for labels, ranking in valid]))


# MRR across queries — excludes 0-relevant topics.
def mean_reciprocal_rank(queries: list[tuple]) -> float:
    """
    Mean Reciprocal Rank over a list of (relevance_labels, ranking) pairs.
    Excludes topics with 0 relevant documents (same TREC standard as MAP).
    """
    valid = _filter_zero_relevant(queries)
    if not valid:
        return 0.0
    return float(np.mean([reciprocal_rank(ranking, labels) for labels, ranking in valid]))


# ---------------------------------------------------------------------------
# PR curves (Lab03 lines 180-244, 438-488)
# ---------------------------------------------------------------------------

# Raw (recall, precision) points at each rank where a relevant doc appears.
def pr_curve(ranking: list[int], relevance: list[bool]) -> tuple[list[float], list[float]]:
    """
    Compute the PR curve for a single query.

    Returns (recalls, precisions) — parallel lists, one point per relevant doc found.
    Only records operating points where a relevant doc is seen (standard PR curve).
    """
    total_relevant = sum(relevance)
    precisions, recalls = [], []
    n_relevant_seen = 0

    for rank, doc_id in enumerate(ranking, start=1):
        if relevance[doc_id]:
            n_relevant_seen += 1
            precisions.append(n_relevant_seen / rank)
            recalls.append(n_relevant_seen / total_relevant)

    return recalls, precisions


# 11-point interpolated PR curve (Lab03 lines 200-213).
def interpolated_pr_curve(
    recalls: list[float],
    precisions: list[float],
    n_points: int = 11,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Standard 11-point interpolation: at each recall level r in {0.0, 0.1, ..., 1.0},
    take the maximum precision at any recall >= r.

    Returns (recall_levels, interp_precisions) as numpy arrays of length n_points.
    """
    recall_levels = np.linspace(0, 1, n_points)
    interp_precisions = []
    for r_level in recall_levels:
        # max precision achieved at recall >= r_level (look-ahead max)
        relevant_precisions = [p for r, p in zip(recalls, precisions) if r >= r_level]
        interp_precisions.append(max(relevant_precisions) if relevant_precisions else 0.0)
    return recall_levels, np.array(interp_precisions)


# Mean interpolated PR curve across queries (Lab03 lines 438-488).
def mean_pr_curve(
    queries: list[tuple],
    n_points: int = 11,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Average interpolated PR curve over multiple (relevance_labels, ranking) pairs.
    Returns (recall_levels, mean_precisions) both of length n_points.
    """
    all_interp = []
    for labels, ranking in queries:
        r, p = pr_curve(ranking, labels)
        _, ip = interpolated_pr_curve(r, p, n_points)
        all_interp.append(ip)

    recall_levels = np.linspace(0, 1, n_points)
    return recall_levels, np.mean(all_interp, axis=0)


# ---------------------------------------------------------------------------
# NDCG — graded relevance (new guide requirement, 2026-03-31)
# ---------------------------------------------------------------------------

# nDCG@k for a single query with graded relevance scores.
def ndcg_at_k(ranking: list[int], relevance_scores: list[float], k: int) -> float:
    """
    Normalised Discounted Cumulative Gain at rank k.

    Args:
        ranking:          list of doc indices, ordered by retrieval rank (best first).
        relevance_scores: list[float] parallel to all_doc_ids — graded relevance (0=not relevant).
        k:                cutoff rank.

    Returns:
        NDCG@k in [0, 1]. Returns 0.0 if no relevant docs exist in ground truth.

    Formula:
        DCG  = sum( rel[i] / log2(i+2)  for i in 0..k-1 )   (i=0 -> log2(2)=1 discount)
        IDCG = DCG of ideal ranking (docs sorted by relevance_scores desc)
        NDCG = DCG / IDCG
    """
    # DCG: iterate top-k results and accumulate discounted gains
    dcg = 0.0
    for i, doc_id in enumerate(ranking[:k]):
        gain = relevance_scores[doc_id]
        discount = np.log2(i + 2)  # log2(2)=1 at rank 1, log2(3) at rank 2, ...
        dcg += gain / discount

    # IDCG: sort all relevance scores descending, take top-k as ideal
    ideal_gains = sorted(relevance_scores, reverse=True)[:k]
    idcg = sum(gain / np.log2(i + 2) for i, gain in enumerate(ideal_gains))

    # avoid divide-by-zero when no relevant docs exist
    if idcg == 0.0:
        return 0.0

    return dcg / idcg


# Mean nDCG@k across queries (excludes 0-relevant topics, same TREC standard as MAP).
def mean_ndcg_at_k(queries: list[tuple], k: int) -> float:
    """
    Mean NDCG@k over a list of (relevance_scores, ranking) pairs.

    Uses the same 0-relevant-topic exclusion as MAP/MRR (TREC standard).
    Queries where sum(relevance_scores) == 0 are skipped and a WARNING is logged.
    """
    valid = []
    n_excluded = 0
    for scores, ranking in queries:
        if sum(scores) == 0:
            n_excluded += 1
        else:
            valid.append((scores, ranking))

    if n_excluded > 0:
        logger.warning(
            "[metrics] NDCG: excluded %d topic(s) with 0 graded relevance (TREC standard).",
            n_excluded,
        )

    if not valid:
        return 0.0

    return float(np.mean([ndcg_at_k(ranking, scores, k) for scores, ranking in valid]))


# ---------------------------------------------------------------------------
# Format conversion (Lab03 lines 756-805)
# ---------------------------------------------------------------------------

def results_to_ranking(
    results: list[tuple[str, float]],
    qrels_set: set[str],
    all_doc_ids: list[str],
) -> tuple[list[bool], list[int]]:
    """
    Convert OpenSearch results to the integer-indexed format expected by metric functions.

    Args:
        results:     list of (doc_id, score) from OpenSearch, ranked by score descending.
        qrels_set:   set of relevant doc_ids for this query.
        all_doc_ids: ordered list of all doc IDs in corpus — defines the index space.

    Returns:
        relevance:  list[bool] parallel to all_doc_ids; True if relevant.
        ranking:    list[int] of indices into all_doc_ids, ordered by retrieval rank.
                    non-retrieved docs are appended at the end (unranked).

    Pattern from Lab03 lines 756-805.
    """
    id_to_idx = {doc_id: i for i, doc_id in enumerate(all_doc_ids)}

    # relevance_labels: parallel to all_doc_ids
    relevance = [doc_id in qrels_set for doc_id in all_doc_ids]

    # build ranking: retrieved ids first (by score), then the rest
    retrieved_ids = [r[0] for r in results]
    retrieved_set = set(retrieved_ids)
    not_retrieved = [doc_id for doc_id in all_doc_ids if doc_id not in retrieved_set]
    full_ranking_ids = retrieved_ids + not_retrieved

    ranking = [id_to_idx[doc_id] for doc_id in full_ranking_ids]

    return relevance, ranking


# Graded version of results_to_ranking — returns float scores instead of bool labels.
# Used for NDCG: relevance_scores[i] = graded score (0/1/2) for the doc at position i.
def results_to_ranking_graded(
    results: list[tuple[str, float]],
    qrels_graded: dict[str, int],
    all_doc_ids: list[str],
) -> tuple[list[float], list[int]]:
    """
    Convert OpenSearch results to graded integer-indexed format for nDCG.

    Args:
        results:       list of (doc_id, score) from OpenSearch, ranked by score descending.
        qrels_graded:  {doc_id: graded_score} — scores are 0/1/2 (from qrels_graded.json for one topic).
        all_doc_ids:   ordered list of all doc IDs in corpus — defines the index space.

    Returns:
        scores:  list[float] parallel to all_doc_ids; 0.0 if not in qrels.
        ranking: list[int] of indices into all_doc_ids, ordered by retrieval rank.
                 non-retrieved docs are appended at the end (unranked).
    """
    id_to_idx = {doc_id: i for i, doc_id in enumerate(all_doc_ids)}

    # graded scores parallel to all_doc_ids
    scores = [float(qrels_graded.get(doc_id, 0)) for doc_id in all_doc_ids]

    # build ranking: retrieved first (by score), then the rest
    retrieved_ids = [r[0] for r in results]
    retrieved_set = set(retrieved_ids)
    not_retrieved = [doc_id for doc_id in all_doc_ids if doc_id not in retrieved_set]
    full_ranking_ids = retrieved_ids + not_retrieved

    ranking = [id_to_idx[doc_id] for doc_id in full_ranking_ids]

    return scores, ranking


# ---------------------------------------------------------------------------
# Self-test: python -m src.evaluation.metrics
# Reproduces the Lab03 toy example to verify correctness.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Step 12 — metrics.py self-test (Lab03 toy example)")
    print("=" * 60)

    # Reproduce Lab03 Part 1 toy corpus (lines 70-83)
    # 10 docs: D0..D9, relevance = [True, False, True, False, True, False, True, False, False, False]
    # 4 relevant docs: D0, D2, D4, D6
    relevance_labels = [True, False, True, False, True, False, True, False, False, False]

    # System A ranks relevant docs early (good system)
    ranking_A = [0, 2, 4, 6, 1, 3, 5, 7, 8, 9]  # all relevant first
    # System B shuffles them more
    ranking_B = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # natural order

    # --- P@k and R@k table ---
    print("\nP@k and R@k comparison:")
    print(f"{'k':>4} | {'P@k (A)':>9} | {'R@k (A)':>9} | {'P@k (B)':>9} | {'R@k (B)':>9}")
    print("-" * 52)
    for k in [1, 2, 3, 5, 7, 10]:
        pa = precision_at_k(ranking_A, relevance_labels, k)
        ra = recall_at_k(ranking_A, relevance_labels, k)
        pb = precision_at_k(ranking_B, relevance_labels, k)
        rb = recall_at_k(ranking_B, relevance_labels, k)
        print(f"{k:>4} | {pa:>9.3f} | {ra:>9.3f} | {pb:>9.3f} | {rb:>9.3f}")

    # --- AP ---
    ap_A = average_precision(ranking_A, relevance_labels)
    ap_B = average_precision(ranking_B, relevance_labels)
    print(f"\nAP (A): {ap_A:.4f}")
    print(f"AP (B): {ap_B:.4f}")

    # System A should have AP = 1.0 (all relevant docs at the top 4 positions)
    assert abs(ap_A - 1.0) < 1e-9, f"Expected AP(A)=1.0, got {ap_A}"
    # System B has relevant docs at ranks 1,3,5,7 -- AP = (1/1 + 2/3 + 3/5 + 4/7) / 4
    expected_ap_B = (1/1 + 2/3 + 3/5 + 4/7) / 4
    assert abs(ap_B - expected_ap_B) < 1e-9, f"Expected AP(B)={expected_ap_B:.4f}, got {ap_B}"
    print(f"  AP assertions passed (A=1.0, B={expected_ap_B:.4f})")

    # --- PR curves ---
    r_A, p_A = pr_curve(ranking_A, relevance_labels)
    r_B, p_B = pr_curve(ranking_B, relevance_labels)
    # System A: recall goes 0.25, 0.5, 0.75, 1.0 with precision 1.0 each time
    assert r_A == [0.25, 0.5, 0.75, 1.0], f"Unexpected recalls_A: {r_A}"
    assert all(abs(p - 1.0) < 1e-9 for p in p_A), f"Unexpected precisions_A: {p_A}"
    print(f"  PR curve A assertions passed")

    # --- 11-point interpolated ---
    ir_A, ip_A = interpolated_pr_curve(r_A, p_A, n_points=11)
    assert len(ir_A) == 11 and len(ip_A) == 11, "Expected 11 interpolation points"
    # System A has perfect precision everywhere
    assert all(abs(p - 1.0) < 1e-9 for p in ip_A), f"Expected all 1.0 for A: {ip_A}"
    print(f"  11-point interpolation assertions passed")

    # --- RR ---
    rr_A = reciprocal_rank(ranking_A, relevance_labels)
    rr_B = reciprocal_rank(ranking_B, relevance_labels)
    assert abs(rr_A - 1.0) < 1e-9, f"Expected RR(A)=1.0, got {rr_A}"
    assert abs(rr_B - 1.0) < 1e-9, f"Expected RR(B)=1.0 (B also has D0 at rank 1), got {rr_B}"
    print(f"  RR assertions passed (both 1.0 — D0 is relevant and rank-1 in both)")

    # --- MAP and MRR on toy multi-query set ---
    # set up 3 fake queries using ranking_A and ranking_B
    queries_A = [(relevance_labels, ranking_A)] * 3
    queries_B = [(relevance_labels, ranking_B)] * 3
    map_A = mean_average_precision(queries_A)
    map_B = mean_average_precision(queries_B)
    mrr_A = mean_reciprocal_rank(queries_A)
    mrr_B = mean_reciprocal_rank(queries_B)
    assert abs(map_A - ap_A) < 1e-9, "MAP(A) should equal AP(A) since all queries identical"
    assert abs(map_B - ap_B) < 1e-9, "MAP(B) should equal AP(B) since all queries identical"
    print(f"  MAP/MRR assertions passed (A: MAP={map_A:.4f} MRR={mrr_A:.4f}, B: MAP={map_B:.4f} MRR={mrr_B:.4f})")

    # --- 0-relevant topic exclusion ---
    import logging
    logging.basicConfig(level=logging.WARNING)
    zero_rel_labels = [False, False, False, False, False, False, False, False, False, False]
    mixed_queries = [(relevance_labels, ranking_A), (zero_rel_labels, ranking_B)]
    map_mixed = mean_average_precision(mixed_queries)
    # MAP should equal AP of first query only (second excluded)
    assert abs(map_mixed - ap_A) < 1e-9, f"Expected MAP={ap_A:.4f} after exclusion, got {map_mixed:.4f}"
    print(f"  0-relevant exclusion assertion passed (MAP = {map_mixed:.4f}, excl. 0-rel query)")

    # --- results_to_ranking ---
    all_ids = ["D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"]
    qrel_set = {"D0", "D2", "D4", "D6"}
    mock_results = [("D0", 0.9), ("D2", 0.8), ("D4", 0.7), ("D6", 0.6)]  # 4 relevant first
    rel, rank = results_to_ranking(mock_results, qrel_set, all_ids)
    assert rel == relevance_labels, f"Relevance labels mismatch: {rel}"
    assert rank[:4] == [0, 2, 4, 6], f"First 4 ranks should be [0,2,4,6], got {rank[:4]}"
    print(f"  results_to_ranking assertions passed")

    # --- mean_pr_curve shape ---
    rl, mp = mean_pr_curve([(relevance_labels, ranking_A), (relevance_labels, ranking_B)])
    assert len(rl) == 11 and len(mp) == 11, f"Expected 11 points, got {len(rl)}"
    print(f"  mean_pr_curve shape assertion passed")

    # ── NDCG tests ─────────────────────────────────────────────────────────
    print("\nNDCG tests:")

    # graded_scores: D0=2, D2=1, D4=2, D6=1, others=0
    # ranking_A retrieves [D0, D2, D4, D6, ...] — all graded docs at top 4
    graded_scores = [2, 0, 1, 0, 2, 0, 1, 0, 0, 0]

    # manual DCG@4 for ranking_A: ranks D0(2), D2(1), D4(2), D6(1)
    # DCG = 2/log2(2) + 1/log2(3) + 2/log2(4) + 1/log2(5)
    import math
    manual_dcg = 2/math.log2(2) + 1/math.log2(3) + 2/math.log2(4) + 1/math.log2(5)
    # ideal: sort [2,2,1,1,0,...] -> DCG = 2/log2(2) + 2/log2(3) + 1/log2(4) + 1/log2(5)
    manual_idcg = 2/math.log2(2) + 2/math.log2(3) + 1/math.log2(4) + 1/math.log2(5)
    expected_ndcg_A = manual_dcg / manual_idcg

    ndcg_A = ndcg_at_k(ranking_A, graded_scores, k=4)
    assert abs(ndcg_A - expected_ndcg_A) < 1e-9, f"Expected nDCG@4(A)={expected_ndcg_A:.4f}, got {ndcg_A:.4f}"
    print(f"  ndcg_at_k assertion passed (nDCG@4(A) = {ndcg_A:.4f})")

    # perfect retrieval: if ranking matches ideal, NDCG should be 1.0
    ideal_ranking = [0, 4, 2, 6, 1, 3, 5, 7, 8, 9]  # D0(2), D4(2), D2(1), D6(1) first
    ndcg_ideal = ndcg_at_k(ideal_ranking, graded_scores, k=4)
    assert abs(ndcg_ideal - 1.0) < 1e-9, f"Expected nDCG@4(ideal)=1.0, got {ndcg_ideal:.4f}"
    print(f"  ndcg_at_k ideal ranking assertion passed (nDCG@4 = {ndcg_ideal:.4f})")

    # zero-relevant topic: NDCG should be 0.0
    zero_scores = [0.0] * 10
    ndcg_zero = ndcg_at_k(ranking_A, zero_scores, k=4)
    assert ndcg_zero == 0.0, f"Expected nDCG=0.0 for zero-relevant, got {ndcg_zero}"
    print(f"  ndcg_at_k zero-relevant assertion passed (nDCG = {ndcg_zero:.4f})")

    # mean_ndcg_at_k: two queries, same ranking_A, one with graded scores one with zeros
    graded_queries = [(graded_scores, ranking_A), (zero_scores, ranking_A)]
    mean_ndcg = mean_ndcg_at_k(graded_queries, k=4)
    # second query excluded (0-relevant) -> mean is just ndcg_A
    assert abs(mean_ndcg - ndcg_A) < 1e-9, f"Expected mean_nDCG={ndcg_A:.4f}, got {mean_ndcg:.4f}"
    print(f"  mean_ndcg_at_k 0-relevant exclusion assertion passed (mean = {mean_ndcg:.4f})")

    # results_to_ranking_graded
    qrels_graded_dict = {"D0": 2, "D2": 1, "D4": 2, "D6": 1}  # matches graded_scores above
    sc, rk = results_to_ranking_graded(mock_results, qrels_graded_dict, all_ids)
    assert sc == [float(x) for x in graded_scores], f"Graded scores mismatch: {sc}"
    assert rk[:4] == [0, 2, 4, 6], f"Graded ranking first 4 should be [0,2,4,6], got {rk[:4]}"
    print(f"  results_to_ranking_graded assertions passed")

    print("\n" + "=" * 60)
    print("metrics.py — all assertions passed")
    print("=" * 60)

