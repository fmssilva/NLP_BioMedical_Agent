"""
src/evaluation/analysis.py

Deeper analysis functions for Phase 1 evaluation:
  - Statistical significance (paired t-test on per-topic AP)
  - Error analysis on hard topics
  - Query length vs AP correlation
  - Document length analysis (relevant vs non-relevant)
  - Strategy agreement / Jaccard overlap
  - IDF analysis of hard vs easy topics
  - Reciprocal rank distribution
  - Confusion matrix at P@10 (using graded qrels)

All functions return data + optional matplotlib figures.
Run standalone: python -m src.evaluation.analysis
"""

import json
import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

PHASE1_DIR = ROOT / "results" / "phase1"
FIGS_DIR   = PHASE1_DIR / "figures"


# ---------------------------------------------------------------------------
# 1. Statistical significance — paired t-test on per-topic AP
# ---------------------------------------------------------------------------

def paired_ttest(
    per_query_a: dict[str, float],
    per_query_b: dict[str, float],
    name_a: str = "A",
    name_b: str = "B",
) -> dict:
    """
    Two-sided paired t-test on per-topic AP values.

    Tests H0: the mean AP difference between system A and B is zero.
    A small p-value (< 0.05) means the difference is statistically significant —
    one system is reliably better, not just lucky on a few topics.

    Returns dict with t-statistic, p-value, mean delta, and interpretation.
    """
    # align by topic id
    common = sorted(set(per_query_a) & set(per_query_b))
    a = np.array([per_query_a[tid] for tid in common])
    b = np.array([per_query_b[tid] for tid in common])

    t_stat, p_val = stats.ttest_rel(a, b)
    delta = float(np.mean(a - b))

    sig = "significant" if p_val < 0.05 else "not significant"
    return {
        "name_a": name_a, "name_b": name_b,
        "n_topics": len(common),
        "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
        "mean_delta": delta,
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "significant_005": bool(p_val < 0.05),
        "interpretation": (
            f"{name_a} vs {name_b}: delta={delta:+.4f}, "
            f"t={t_stat:.3f}, p={p_val:.4f} ({sig} at alpha=0.05)"
        ),
    }


def run_significance_tests(
    all_per_query: dict[str, dict[str, float]],
) -> list[dict]:
    """
    Run paired t-tests between all interesting strategy pairs.

    Args:
        all_per_query: {strategy_name: {topic_id: AP}} for each strategy.

    Returns list of test result dicts from paired_ttest().
    """
    names = list(all_per_query.keys())
    tests = []
    # compare every pair once
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            tests.append(paired_ttest(
                all_per_query[names[i]], all_per_query[names[j]],
                names[i], names[j],
            ))
    return tests


def print_significance_table(tests: list[dict]) -> None:
    """Print a clean summary table of all paired t-tests."""
    print(f"\n{'=' * 80}")
    print("Paired t-test results (two-sided, per-topic AP)")
    print(f"{'=' * 80}")
    print(f"{'Comparison':>40} | {'Delta':>8} | {'t-stat':>8} | {'p-value':>8} | {'Sig?':>5}")
    print("-" * 80)
    for t in tests:
        label = f"{t['name_a']} vs {t['name_b']}"
        sig = "YES" if t["significant_005"] else "no"
        print(f"{label:>40} | {t['mean_delta']:>+8.4f} | {t['t_statistic']:>8.3f} | "
              f"{t['p_value']:>8.4f} | {sig:>5}")
    print("-" * 80)


# ---------------------------------------------------------------------------
# 2. Error analysis on hard topics
# ---------------------------------------------------------------------------

def error_analysis(
    per_query: dict[str, dict],
    topics: list[dict],
    qrels: dict,
    n_hard: int = 5,
) -> dict:
    """
    Identify the hardest and easiest topics by AP. For each, show the query
    text, number of relevant docs, and AP across strategies that were evaluated.

    Args:
        per_query: {strategy: {tid: {"AP": float, ...}}} from final_eval results.
        topics: list of topic dicts with id, topic, question, narrative.
        qrels: binary qrels {tid: {docid: rel}}.
        n_hard: how many worst/best topics to show.

    Returns dict with "hardest" and "easiest" topic lists.
    """
    topic_map = {str(t["id"]): t for t in topics}
    strategies = list(per_query.keys())

    # mean AP across all strategies for each topic
    topic_ids = sorted(set().union(*(pq.keys() for pq in per_query.values())))
    mean_aps = {}
    for tid in topic_ids:
        aps = [per_query[s].get(tid, {}).get("AP", 0.0) for s in strategies if tid in per_query[s]]
        mean_aps[tid] = float(np.mean(aps)) if aps else 0.0

    sorted_topics = sorted(mean_aps.items(), key=lambda x: x[1])
    hardest = sorted_topics[:n_hard]
    easiest = sorted_topics[-n_hard:][::-1]

    def _detail(tid_ap_list):
        details = []
        for tid, mean_ap in tid_ap_list:
            t = topic_map.get(tid, {})
            n_rel = len(qrels.get(tid, {}))
            per_strat = {s: per_query[s].get(tid, {}).get("AP", None) for s in strategies}
            details.append({
                "topic_id": tid,
                "mean_AP": mean_ap,
                "n_relevant": n_rel,
                "topic_text": t.get("topic", ""),
                "question": t.get("question", ""),
                "per_strategy_AP": per_strat,
            })
        return details

    return {"hardest": _detail(hardest), "easiest": _detail(easiest)}


def print_error_analysis(analysis: dict) -> None:
    """Print hard/easy topic analysis in readable format."""
    for label, key in [("HARDEST", "hardest"), ("EASIEST", "easiest")]:
        print(f"\n--- {label} topics (by mean AP across strategies) ---")
        for d in analysis[key]:
            print(f"\n  Topic {d['topic_id']}  (mean AP={d['mean_AP']:.4f}, {d['n_relevant']} relevant docs)")
            print(f"    Topic: {d['topic_text']}")
            print(f"    Question: {d['question']}")
            for s, ap in d["per_strategy_AP"].items():
                if ap is not None:
                    print(f"      {s:>25}: AP={ap:.4f}")


# ---------------------------------------------------------------------------
# 3. Query length vs AP scatter
# ---------------------------------------------------------------------------

def plot_query_length_vs_ap(
    per_query_ap: dict[str, dict[str, float]],
    topics: list[dict],
    strategy_name: str = "RRF (tuned)",
    save_path: str | None = None,
) -> tuple[plt.Figure, float, float]:
    """
    Scatter plot of query word count vs AP. Returns (fig, pearson_r, p_value).

    Tests hypothesis: do longer concatenated queries (topic+question+narrative)
    tend to have higher AP?
    """
    topic_map = {str(t["id"]): t for t in topics}
    ap_data = per_query_ap.get(strategy_name, {})

    tids, lengths, aps = [], [], []
    for tid, ap in ap_data.items():
        t = topic_map.get(tid, {})
        # word count of concatenated query
        concat = f"{t.get('topic', '')} {t.get('question', '')} {t.get('narrative', '')}"
        wc = len(concat.split())
        tids.append(tid)
        lengths.append(wc)
        aps.append(ap)

    lengths = np.array(lengths)
    aps = np.array(aps)

    r, p_val = stats.pearsonr(lengths, aps) if len(lengths) > 2 else (0.0, 1.0)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(lengths, aps, s=40, alpha=0.7, color="steelblue", edgecolors="white", lw=0.5)

    # trend line
    if len(lengths) > 2:
        z = np.polyfit(lengths, aps, 1)
        x_line = np.linspace(lengths.min(), lengths.max(), 50)
        ax.plot(x_line, np.polyval(z, x_line), "--", color="tomato", lw=1.5,
                label=f"r={r:.3f}, p={p_val:.3f}")

    # annotate a few extreme points
    for i in [np.argmin(aps), np.argmax(aps)]:
        ax.annotate(f"T{tids[i]}", (lengths[i], aps[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7, alpha=0.8)

    ax.set_xlabel("Query word count (concatenated)")
    ax.set_ylabel("Average Precision")
    ax.set_title(f"Query Length vs AP — {strategy_name}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120)
        print(f"[analysis] Saved query-length plot -> {save_path}")
    return fig, float(r), float(p_val)


# ---------------------------------------------------------------------------
# 4. Document length analysis (relevant vs non-relevant)
# ---------------------------------------------------------------------------

def plot_doc_length_distribution(
    corpus: list[dict],
    qrels: dict,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Histogram of document word counts, split by relevant vs non-relevant.
    Relevant = appears in qrels for any topic.

    Tests whether relevant documents tend to be longer or shorter,
    which helps explain why b=1.0 (full length normalisation) helps.
    """
    rel_ids = set()
    for tid_rels in qrels.values():
        rel_ids.update(tid_rels.keys())

    rel_lengths, nonrel_lengths = [], []
    for doc in corpus:
        wc = len(doc.get("contents", "").split())
        if doc["id"] in rel_ids:
            rel_lengths.append(wc)
        else:
            nonrel_lengths.append(wc)

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, max(max(rel_lengths, default=0), max(nonrel_lengths, default=0)), 40)
    ax.hist(nonrel_lengths, bins=bins, alpha=0.5, color="steelblue", label=f"Non-relevant ({len(nonrel_lengths)})")
    ax.hist(rel_lengths, bins=bins, alpha=0.6, color="tomato", label=f"Relevant ({len(rel_lengths)})")

    ax.axvline(np.median(rel_lengths), color="tomato", ls="--", lw=1.5,
               label=f"Relevant median={np.median(rel_lengths):.0f}")
    ax.axvline(np.median(nonrel_lengths), color="steelblue", ls="--", lw=1.5,
               label=f"Non-relevant median={np.median(nonrel_lengths):.0f}")

    ax.set_xlabel("Document word count")
    ax.set_ylabel("Frequency")
    ax.set_title("Document Length: Relevant vs Non-Relevant")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120)
        print(f"[analysis] Saved doc-length plot -> {save_path}")
    return fig


# ---------------------------------------------------------------------------
# 5. Strategy agreement / Jaccard overlap
# ---------------------------------------------------------------------------

def jaccard_overlap(
    runs: dict[str, dict[str, list]],
    k: int = 10,
) -> dict[tuple[str, str], float]:
    """
    Compute mean Jaccard overlap of top-k results between all strategy pairs,
    averaged across topics.

    Returns {(nameA, nameB): mean_jaccard}.
    """
    names = list(runs.keys())
    # align topic IDs
    all_tids = sorted(set.intersection(*(set(runs[n].keys()) for n in names)))

    results = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            jaccards = []
            for tid in all_tids:
                set_a = set(d[0] if isinstance(d, (list, tuple)) else d
                            for d in runs[names[i]][tid][:k])
                set_b = set(d[0] if isinstance(d, (list, tuple)) else d
                            for d in runs[names[j]][tid][:k])
                union = set_a | set_b
                inter = set_a & set_b
                jaccards.append(len(inter) / len(union) if union else 0.0)
            results[(names[i], names[j])] = float(np.mean(jaccards))
    return results


def plot_jaccard_heatmap(
    overlaps: dict[tuple[str, str], float],
    save_path: str | None = None,
) -> plt.Figure:
    """Heatmap of pairwise Jaccard overlap at top-10."""
    names = sorted(set(n for pair in overlaps for n in pair))
    n = len(names)
    mat = np.ones((n, n))  # diagonal = 1
    name_idx = {name: i for i, name in enumerate(names)}
    for (a, b), val in overlaps.items():
        mat[name_idx[a]][name_idx[b]] = val
        mat[name_idx[b]][name_idx[a]] = val

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    # shorten names for readability
    short = [name.replace(" (tuned)", "*").replace(" (default)", "") for name in names]
    ax.set_xticklabels(short, fontsize=8, rotation=45, ha="right")
    ax.set_yticklabels(short, fontsize=8)

    # annotate cells
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if mat[i, j] > 0.5 else "black")

    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Pairwise Jaccard Overlap (top-10)")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120)
        print(f"[analysis] Saved Jaccard heatmap -> {save_path}")
    return fig


# ---------------------------------------------------------------------------
# 6. IDF analysis of hard vs easy topics
# ---------------------------------------------------------------------------

def idf_analysis(
    topics: list[dict],
    corpus: list[dict],
    per_query_ap: dict[str, float],
    n_hard: int = 5,
    n_easy: int = 5,
) -> dict:
    """
    Compute mean IDF of query terms for hard vs easy topics.

    IDF(t) = log(N / df(t)) where df(t) = # docs containing term t.
    High IDF = rare term (discriminative), low IDF = common term.

    Returns dict with hard_topics, easy_topics, and their mean IDFs.
    """
    N = len(corpus)
    # build df lookup
    doc_freq = {}
    for doc in corpus:
        words = set(doc.get("contents", "").lower().split())
        for w in words:
            doc_freq[w] = doc_freq.get(w, 0) + 1

    def mean_idf(text: str) -> float:
        words = text.lower().split()
        if not words:
            return 0.0
        idfs = []
        for w in words:
            df = doc_freq.get(w, 0)
            idfs.append(np.log(N / (df + 1)))  # +1 smoothing
        return float(np.mean(idfs))

    topic_map = {str(t["id"]): t for t in topics}
    sorted_ap = sorted(per_query_ap.items(), key=lambda x: x[1])
    hard = sorted_ap[:n_hard]
    easy = sorted_ap[-n_easy:]

    def _detail(items):
        details = []
        for tid, ap in items:
            t = topic_map.get(tid, {})
            concat = f"{t.get('topic', '')} {t.get('question', '')} {t.get('narrative', '')}"
            details.append({
                "topic_id": tid,
                "AP": ap,
                "mean_idf": mean_idf(concat),
                "query": t.get("topic", ""),
            })
        return details

    hard_details = _detail(hard)
    easy_details = _detail(easy)
    return {
        "hard": hard_details,
        "easy": easy_details,
        "mean_idf_hard": float(np.mean([d["mean_idf"] for d in hard_details])),
        "mean_idf_easy": float(np.mean([d["mean_idf"] for d in easy_details])),
    }


def print_idf_analysis(result: dict) -> None:
    print(f"\n--- IDF Analysis: Hard vs Easy Topics ---")
    print(f"  Mean IDF (hard topics): {result['mean_idf_hard']:.3f}")
    print(f"  Mean IDF (easy topics): {result['mean_idf_easy']:.3f}")
    for label, key in [("Hard", "hard"), ("Easy", "easy")]:
        print(f"\n  {label} topics:")
        for d in result[key]:
            print(f"    T{d['topic_id']}: AP={d['AP']:.4f}, mean_idf={d['mean_idf']:.3f} — \"{d['query']}\"")


# ---------------------------------------------------------------------------
# 7. Reciprocal rank distribution
# ---------------------------------------------------------------------------

def plot_rr_distribution(
    per_query_rr: dict[str, dict[str, float]],
    strategy_name: str = "RRF (tuned)",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Histogram of reciprocal ranks across topics for a strategy.
    RR=1 means the first result was relevant, RR=0.5 means rank 2, etc.
    """
    rr_vals = list(per_query_rr.get(strategy_name, {}).values())

    fig, ax = plt.subplots(figsize=(7, 4))
    # custom bins at 1, 0.5, 0.33, 0.25, 0.2, 0.1, 0
    bin_edges = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.33, 0.5, 1.01]
    ax.hist(rr_vals, bins=bin_edges, color="steelblue", edgecolor="white", alpha=0.8)

    # annotate rank equivalences
    rank_labels = {1.0: "rank 1", 0.5: "rank 2", 0.33: "rank 3", 0.25: "rank 4", 0.2: "rank 5"}
    for rr, label in rank_labels.items():
        count = sum(1 for v in rr_vals if abs(v - rr) < 0.01)
        if count > 0:
            ax.annotate(f"{label}\n(n={count})", xy=(rr, count),
                        textcoords="offset points", xytext=(0, 10),
                        fontsize=7, ha="center", alpha=0.8)

    mrr = float(np.mean(rr_vals)) if rr_vals else 0
    ax.axvline(mrr, color="tomato", ls="--", lw=1.5, label=f"MRR = {mrr:.3f}")

    ax.set_xlabel("Reciprocal Rank")
    ax.set_ylabel("Number of topics")
    ax.set_title(f"RR Distribution — {strategy_name}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120)
        print(f"[analysis] Saved RR distribution -> {save_path}")
    return fig


# ---------------------------------------------------------------------------
# 8. Confusion matrix at P@10 (graded qrels)
# ---------------------------------------------------------------------------

def confusion_at_k(
    runs: dict[str, dict[str, list]],
    qrels_graded: dict,
    k: int = 10,
) -> dict[str, dict[str, int]]:
    """
    For each strategy, count top-k docs by graded relevance category.

    Categories from BioGen graded qrels:
      - supporting (score=2): directly relevant
      - neutral (score=1): partially relevant / background
      - not relevant (score=0 or not in qrels): irrelevant

    Returns {strategy: {"supporting": int, "neutral": int, "not_relevant": int}}.
    """
    results = {}
    for name, run in runs.items():
        counts = {"supporting": 0, "neutral": 0, "not_relevant": 0}
        n_topics = 0
        for tid, docs in run.items():
            n_topics += 1
            graded = qrels_graded.get(tid, {})
            for doc_entry in docs[:k]:
                doc_id = doc_entry[0] if isinstance(doc_entry, (list, tuple)) else doc_entry
                score = graded.get(doc_id, 0)
                if score >= 2:
                    counts["supporting"] += 1
                elif score >= 1:
                    counts["neutral"] += 1
                else:
                    counts["not_relevant"] += 1
        results[name] = counts
    return results


def plot_confusion_bars(
    confusion: dict[str, dict[str, int]],
    k: int = 10,
    save_path: str | None = None,
) -> plt.Figure:
    """Stacked bar chart showing composition of top-k results by relevance grade."""
    names = list(confusion.keys())
    supporting = [confusion[n]["supporting"] for n in names]
    neutral    = [confusion[n]["neutral"]    for n in names]
    not_rel    = [confusion[n]["not_relevant"] for n in names]

    fig, ax = plt.subplots(figsize=(max(7, len(names) * 1.3), 5))
    x = np.arange(len(names))
    width = 0.5

    ax.bar(x, supporting, width, label="Supporting (2)", color="seagreen", alpha=0.85)
    ax.bar(x, neutral, width, bottom=supporting, label="Neutral (1)", color="goldenrod", alpha=0.85)
    bottoms = [s + n for s, n in zip(supporting, neutral)]
    ax.bar(x, not_rel, width, bottom=bottoms, label="Not relevant (0)", color="tomato", alpha=0.85)

    # total label on top
    for i, name in enumerate(names):
        total = supporting[i] + neutral[i] + not_rel[i]
        ax.text(i, total + 1, f"{supporting[i]}/{total}", ha="center", fontsize=8)

    ax.set_xticks(x)
    short = [name.replace(" (tuned)", "*").replace(" (default)", "") for name in names]
    ax.set_xticklabels(short, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("Document count")
    ax.set_title(f"Top-{k} Composition by Relevance Grade")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120)
        print(f"[analysis] Saved confusion bars -> {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Entry point: run all analyses on saved final_eval results
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    from dotenv import load_dotenv
    load_dotenv()

    from src.data.loader import load_corpus, load_topics
    from src.evaluation.evaluator import load_run

    print("=" * 78)
    print("Phase 1 — Deeper Analysis")
    print("=" * 78)

    corpus = load_corpus(ROOT / "data" / "filtered_pubmed_abstracts.txt")

    with open(ROOT / "results" / "splits" / "test_queries.json") as f:
        test_topics = json.load(f)
    with open(ROOT / "results" / "qrels.json") as f:
        qrels = json.load(f)
    with open(ROOT / "results" / "qrels_graded.json") as f:
        qrels_graded = json.load(f)

    # load the final eval summary
    with open(PHASE1_DIR / "final_eval_summary.json") as f:
        summary = json.load(f)

    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    # -- load all run files --
    run_files = {
        "BM25 (default)":  PHASE1_DIR / "bm25_run.json",
        "BM25 (tuned)":    PHASE1_DIR / "bm25_tuned_run.json",
        "LM-Dir (mu=75)":  PHASE1_DIR / "lm-dir_mu75_run.json",
        "KNN (MedCPT)":    PHASE1_DIR / "knn_medcpt_run.json",
        "RRF (tuned)":     PHASE1_DIR / "rrf_tuned_run.json",
    }
    runs = {}
    for name, path in run_files.items():
        if path.exists():
            runs[name] = load_run(path)
        else:
            print(f"  [skip] {name} — run file not found: {path}")

    # -- recompute per-query AP from runs (for significance tests) --
    from src.evaluation.metrics import (
        average_precision, reciprocal_rank,
        results_to_ranking, results_to_ranking_graded,
    )
    all_doc_ids = [doc["id"] for doc in corpus]

    per_query_ap = {}
    per_query_rr = {}
    for name, run in runs.items():
        pq_ap, pq_rr = {}, {}
        for tid, results in run.items():
            qrels_set = set(qrels.get(tid, {}).keys())
            # results may be list of [pmid, score] from JSON
            results_tuples = [(r[0], r[1]) for r in results]
            relevance, ranking = results_to_ranking(results_tuples, qrels_set, all_doc_ids)
            pq_ap[tid] = average_precision(ranking, relevance)
            pq_rr[tid] = reciprocal_rank(ranking, relevance)
        per_query_ap[name] = pq_ap
        per_query_rr[name] = pq_rr

    # =====================================================================
    # 1. Statistical significance
    # =====================================================================
    print("\n--- 1. Statistical Significance (paired t-test) ---")
    tests = run_significance_tests(per_query_ap)
    print_significance_table(tests)

    # save
    with open(PHASE1_DIR / "significance_tests.json", "w") as f:
        json.dump(tests, f, indent=2)

    # =====================================================================
    # 2. Error analysis
    # =====================================================================
    print("\n--- 2. Error Analysis on Hard/Easy Topics ---")
    per_query_full = {name: {tid: {"AP": ap} for tid, ap in pq.items()} for name, pq in per_query_ap.items()}
    ea = error_analysis(per_query_full, test_topics, qrels, n_hard=5)
    print_error_analysis(ea)

    with open(PHASE1_DIR / "error_analysis.json", "w") as f:
        json.dump(ea, f, indent=2)

    # =====================================================================
    # 3. Query length vs AP
    # =====================================================================
    print("\n--- 3. Query Length vs AP ---")
    best_strat = "RRF (tuned)" if "RRF (tuned)" in per_query_ap else list(per_query_ap.keys())[-1]
    fig_ql, r, p = plot_query_length_vs_ap(
        {name: pq for name, pq in per_query_ap.items()},
        test_topics, best_strat,
        save_path=str(FIGS_DIR / "query_length_vs_ap.png"),
    )
    plt.close(fig_ql)
    print(f"  Pearson r={r:.3f}, p={p:.4f}")

    # =====================================================================
    # 4. Document length distribution
    # =====================================================================
    print("\n--- 4. Document Length Distribution ---")
    fig_dl = plot_doc_length_distribution(
        corpus, qrels,
        save_path=str(FIGS_DIR / "doc_length_distribution.png"),
    )
    plt.close(fig_dl)

    # =====================================================================
    # 5. Strategy agreement / Jaccard overlap
    # =====================================================================
    print("\n--- 5. Strategy Agreement (Jaccard at top-10) ---")
    overlaps = jaccard_overlap(runs, k=10)
    for (a, b), j in sorted(overlaps.items(), key=lambda x: -x[1]):
        short_a = a.replace(" (tuned)", "*").replace(" (default)", "")
        short_b = b.replace(" (tuned)", "*").replace(" (default)", "")
        print(f"  {short_a:>18} vs {short_b:<18}: J={j:.3f}")

    fig_j = plot_jaccard_heatmap(
        overlaps,
        save_path=str(FIGS_DIR / "jaccard_overlap.png"),
    )
    plt.close(fig_j)

    # =====================================================================
    # 6. IDF analysis
    # =====================================================================
    print("\n--- 6. IDF Analysis: Hard vs Easy Topics ---")
    idf = idf_analysis(test_topics, corpus, per_query_ap.get(best_strat, {}))
    print_idf_analysis(idf)

    with open(PHASE1_DIR / "idf_analysis.json", "w") as f:
        json.dump(idf, f, indent=2)

    # =====================================================================
    # 7. RR distribution
    # =====================================================================
    print("\n--- 7. Reciprocal Rank Distribution ---")
    fig_rr = plot_rr_distribution(
        per_query_rr, best_strat,
        save_path=str(FIGS_DIR / "rr_distribution.png"),
    )
    plt.close(fig_rr)

    # =====================================================================
    # 8. Confusion matrix at P@10
    # =====================================================================
    print("\n--- 8. Confusion Matrix at P@10 ---")
    confusion = confusion_at_k(runs, qrels_graded, k=10)
    for name, c in confusion.items():
        total = sum(c.values())
        print(f"  {name:>20}: {c['supporting']} supporting, {c['neutral']} neutral, "
              f"{c['not_relevant']} irrelevant  ({c['supporting']}/{total} = "
              f"{c['supporting']/total:.1%} precision)")

    fig_cm = plot_confusion_bars(
        confusion, k=10,
        save_path=str(FIGS_DIR / "confusion_at_p10.png"),
    )
    plt.close(fig_cm)

    print("\n" + "=" * 78)
    print(f"All analysis complete. Figures saved to: {FIGS_DIR}")
    print("=" * 78)
