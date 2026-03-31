"""
src/evaluation/plots.py

Visualisation functions for Phase 1 retrieval evaluation.
All plots follow Lab03 style (lines 438-488, 811-855).

Public API:
    plot_pr_comparison(strategy_curves, title, save_path)              -- mean PR curves, all strategies
    plot_metric_table(metric_dict, save_path)                          -- MAP/MRR bar chart
    plot_per_topic_variance(per_topic_ap, save_path)                   -- AP box plot per strategy
    plot_individual_pr_curves(per_query_curves, ap_scores,             -- 3 highlighted individual PR curves
                              highlight_ids, title, save_path)

`strategy_curves` format: {strategy_name: (recall_levels, mean_precisions)}
`metric_dict`     format: {strategy_name: {"MAP": float, "MRR": float, "P@10": float}}
`per_topic_ap`    format: {strategy_name: [ap_q1, ap_q2, ...]}
`per_query_curves` format: {topic_id: (recalls, precisions)}    -- raw (not interpolated) PR points
`highlight_ids`   format: {"best": topic_id, "worst": topic_id, "extra": topic_id}

Lab03 reference: Lab03_Retrieval_Evaluation.ipynb lines 438-488, 811-855.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# consistent colors for all 5 strategies — same order always
_STRATEGY_COLORS = {
    "BM25":      "steelblue",
    "LM-JM":     "tomato",
    "LM-Dir":    "seagreen",
    "KNN":       "darkorange",
    "RRF":       "purple",
}

# fallback palette for strategies not in the map above
_FALLBACK_PALETTE = plt.cm.tab10(np.linspace(0, 0.9, 10))


def _get_color(name: str, idx: int) -> str:
    """Return a consistent color for a strategy name, or a fallback by index."""
    for key, color in _STRATEGY_COLORS.items():
        if key.lower() in name.lower():
            return color
    return _FALLBACK_PALETTE[idx % len(_FALLBACK_PALETTE)]


# --- Plot 1: mean PR curves, all strategies on same axes -------------------

def plot_pr_comparison(
    strategy_curves: dict[str, tuple],
    title: str = "Mean Interpolated PR Curves",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot mean interpolated PR curve for each strategy on the same axes.
    Includes fill_between shading and MAP in the legend.

    Args:
        strategy_curves: {name: (recall_levels, mean_precisions)} — 11-point arrays.
        title:           plot title.
        save_path:       if given, save to this path instead of showing.

    Returns:
        The matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for idx, (name, (rl, mp)) in enumerate(strategy_curves.items()):
        color = _get_color(name, idx)
        # MAP is the mean of the 11-point precision values (approximate, enough for legend)
        map_approx = float(np.mean(mp))
        ax.plot(rl, mp, "o-", color=color, lw=2, ms=4,
                label=f"{name}  (MAP≈{map_approx:.3f})")
        ax.fill_between(rl, mp, alpha=0.08, color=color)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.10])
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120)
        print(f"[plots] Saved PR comparison -> {save_path}")
    return fig


# --- Plot 2: MAP/MRR bar chart, side by side per strategy ------------------

def plot_metric_table(
    metric_dict: dict[str, dict[str, float]],
    metrics_to_plot: list[str] | None = None,
    title: str = "Metric Comparison",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Bar chart comparing MAP and MRR (and optionally P@10) for all strategies.
    Value labels are placed on top of each bar (Lab03 lines 473-484).

    Args:
        metric_dict:     {strategy_name: {"MAP": float, "MRR": float, "P@10": float}}.
        metrics_to_plot: list of metric keys to include, default ["MAP", "MRR"].
        title:           plot title.
        save_path:       if given, save to this path.

    Returns:
        The matplotlib Figure object.
    """
    if metrics_to_plot is None:
        metrics_to_plot = ["MAP", "MRR"]

    strategy_names = list(metric_dict.keys())
    x = np.arange(len(metrics_to_plot))
    width = 0.8 / len(strategy_names)  # divide bar width evenly

    fig, ax = plt.subplots(figsize=(max(8, len(strategy_names) * 2), 5))

    for idx, (name, vals) in enumerate(metric_dict.items()):
        color = _get_color(name, idx)
        offsets = x - (len(strategy_names) - 1) * width / 2 + idx * width
        bars = ax.bar(offsets, [vals.get(m, 0.0) for m in metrics_to_plot],
                      width, label=name, color=color, alpha=0.85)
        # value labels on top of each bar — Lab03 pattern
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot)
    ax.set_ylim([0, 1.05])
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120)
        print(f"[plots] Saved metric bar chart -> {save_path}")
    return fig


# --- Plot 3: per-topic AP variance box plot --------------------------------

def plot_per_topic_variance(
    per_topic_ap: dict[str, list[float]],
    title: str = "Per-Topic AP Distribution",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Box plot showing the distribution of per-topic AP scores for each strategy.
    Reveals consistency: wide boxes = high variability across topics.

    Args:
        per_topic_ap: {strategy_name: [ap_q1, ap_q2, ...]} — one value per test query.
        title:        plot title.
        save_path:    if given, save to this path.

    Returns:
        The matplotlib Figure object.
    """
    strategy_names = list(per_topic_ap.keys())
    data = [per_topic_ap[name] for name in strategy_names]

    fig, ax = plt.subplots(figsize=(max(8, len(strategy_names) * 1.5), 5))

    bp = ax.boxplot(data, patch_artist=True, notch=False, vert=True)

    # color each box to match the strategy palette
    for idx, (patch, name) in enumerate(zip(bp["boxes"], strategy_names)):
        patch.set_facecolor(_get_color(name, idx))
        patch.set_alpha(0.6)

    # median line in white for contrast
    for median in bp["medians"]:
        median.set_color("white")
        median.set_linewidth(2)

    ax.set_xticks(range(1, len(strategy_names) + 1))
    ax.set_xticklabels(strategy_names, fontsize=10)
    ax.set_ylabel("Average Precision")
    ax.set_ylim([-0.05, 1.05])
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120)
        print(f"[plots] Saved AP box plot -> {save_path}")
    return fig


# --- Combined figure: PR curves + metric table side by side ----------------

def plot_combined(
    strategy_curves: dict[str, tuple],
    metric_dict: dict[str, dict[str, float]],
    title: str = "Phase 1 Retrieval Evaluation",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Two-panel figure (Lab03 style): left = mean PR curves, right = MAP/MRR bar chart.
    Useful for notebook / report inclusion as a single image.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # left: PR curves
    ax = axes[0]
    for idx, (name, (rl, mp)) in enumerate(strategy_curves.items()):
        color = _get_color(name, idx)
        map_approx = float(np.mean(mp))
        ax.plot(rl, mp, "o-", color=color, lw=2, ms=4,
                label=f"{name}  (MAP≈{map_approx:.3f})")
        ax.fill_between(rl, mp, alpha=0.08, color=color)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Mean Interpolated PR Curves")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.10])
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # right: bar chart
    ax = axes[1]
    strategy_names = list(metric_dict.keys())
    metrics_to_plot = ["MAP", "MRR"]
    x = np.arange(len(metrics_to_plot))
    width = 0.8 / len(strategy_names)

    for idx, (name, vals) in enumerate(metric_dict.items()):
        color = _get_color(name, idx)
        offsets = x - (len(strategy_names) - 1) * width / 2 + idx * width
        bars = ax.bar(offsets, [vals.get(m, 0.0) for m in metrics_to_plot],
                      width, label=name, color=color, alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot)
    ax.set_ylim([0, 1.05])
    ax.set_title("MAP & MRR Comparison")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120)
        print(f"[plots] Saved combined figure -> {save_path}")
    return fig


# --- Plot 5: individual per-query PR curves for selected topics -------------

def plot_individual_pr_curves(
    per_query_curves: dict[str, tuple],
    ap_scores: dict[str, float],
    highlight_ids: dict[str, str],
    strategy_name: str = "",
    title: str = "Individual PR Curves",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot individual (raw, non-interpolated) PR curves for all queries in a strategy.

    The 3 highlighted queries are drawn in distinct colors with labeled AP scores.
    All other queries are drawn in light gray as background to give context.

    Required by new project guide (2026-03-31):
    "At least three specific query precision-recall curves should be discussed:
     the highest AP query, the lowest AP query, and one additional."

    Args:
        per_query_curves: {topic_id: (recalls, precisions)} — raw PR points per query.
        ap_scores:        {topic_id: ap_float} — AP value for each query (for legend labels).
        highlight_ids:    {"best": topic_id, "worst": topic_id, "extra": topic_id}.
        strategy_name:    name of the retrieval strategy (for the title).
        title:            plot title.
        save_path:        if given, save to this path.

    Returns:
        The matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # highlight colors — best=green, worst=red, extra=orange
    highlight_colors = {
        "best":  "seagreen",
        "worst": "tomato",
        "extra": "darkorange",
    }
    highlight_labels = {
        "best":  "Best AP",
        "worst": "Worst AP",
        "extra": "Middle AP",
    }

    # all highlighted topic ids for fast lookup
    highlighted = set(highlight_ids.values())

    # draw all non-highlighted curves first — thin gray background
    for topic_id, (recalls, precisions) in per_query_curves.items():
        if topic_id not in highlighted:
            ax.plot(recalls, precisions, color="lightgray", lw=0.8, alpha=0.5, zorder=1)

    # draw highlighted curves on top
    for role, topic_id in highlight_ids.items():
        if topic_id not in per_query_curves:
            continue
        recalls, precisions = per_query_curves[topic_id]
        color = highlight_colors.get(role, "black")
        ap = ap_scores.get(topic_id, 0.0)
        label = f"{highlight_labels.get(role, role)} — topic {topic_id}  (AP={ap:.3f})"
        # step-style line shows the actual staircase shape of a PR curve
        ax.step(recalls, precisions, where="post", color=color, lw=2.2, zorder=3, label=label)
        # add dots at each actual operating point
        ax.scatter(recalls, precisions, color=color, s=20, zorder=4)

    # also draw the mAP reference line from ap_scores
    if ap_scores:
        mean_ap = float(np.mean(list(ap_scores.values())))
        ax.axhline(mean_ap, color="navy", linestyle="--", lw=1.2, alpha=0.7,
                   label=f"mAP = {mean_ap:.3f}  (all {len(ap_scores)} queries)", zorder=2)

    title_str = f"{title}" + (f" — {strategy_name}" if strategy_name else "")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title_str)
    ax.set_xlim([-0.02, 1.05])
    ax.set_ylim([-0.02, 1.05])
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120)
        print(f"[plots] Saved individual PR curves -> {save_path}")
    return fig



if __name__ == "__main__":
    print("=" * 60)
    print("Step 13 — plots.py self-test (dummy data, no crash check)")
    print("=" * 60)

    import tempfile

    # dummy data — 5 strategies, 11 recall levels, fabricated PR curves
    np.random.seed(42)
    rl = np.linspace(0, 1, 11)
    strategy_names = ["BM25", "LM-JM", "LM-Dir", "KNN", "RRF"]
    # RRF is best, others spread around
    base_perf = [0.35, 0.33, 0.30, 0.28, 0.42]

    strategy_curves = {}
    metric_dict = {}
    per_topic_ap = {}

    for name, perf in zip(strategy_names, base_perf):
        mp = np.maximum(0, perf - rl * perf + np.random.uniform(-0.02, 0.02, 11))
        strategy_curves[name] = (rl, mp)
        map_val = float(np.mean(mp))
        metric_dict[name] = {"MAP": map_val, "MRR": map_val + 0.1, "P@10": map_val + 0.05}
        # 33 fake per-topic APs
        per_topic_ap[name] = list(np.clip(np.random.normal(map_val, 0.15, 33), 0, 1))

    # temp dir for output files
    with tempfile.TemporaryDirectory() as tmpdir:
        p1 = os.path.join(tmpdir, "pr_comparison.png")
        p2 = os.path.join(tmpdir, "metric_table.png")
        p3 = os.path.join(tmpdir, "ap_boxplot.png")
        p4 = os.path.join(tmpdir, "combined.png")

        fig1 = plot_pr_comparison(strategy_curves, save_path=p1)
        plt.close(fig1)
        print(f"  plot_pr_comparison: OK ({p1})")

        fig2 = plot_metric_table(metric_dict, save_path=p2)
        plt.close(fig2)
        print(f"  plot_metric_table: OK ({p2})")

        fig3 = plot_per_topic_variance(per_topic_ap, save_path=p3)
        plt.close(fig3)
        print(f"  plot_per_topic_variance: OK ({p3})")

        fig4 = plot_combined(strategy_curves, metric_dict, save_path=p4)
        plt.close(fig4)
        print(f"  plot_combined: OK ({p4})")

        # verify files were created
        for p in [p1, p2, p3, p4]:
            assert os.path.exists(p), f"File not created: {p}"
        print("  All plot files created and verified.")

        # --- plot_individual_pr_curves test ---
        # fake per-query curves: 33 queries, each with a short raw PR curve
        np.random.seed(99)
        per_query_curves = {}
        ap_scores_test = {}
        for i in range(33):
            qid = str(116 + i * 2)
            # generate 3-5 operating points per query (raw PR curve)
            n_pts = np.random.randint(3, 8)
            raw_recalls = np.sort(np.random.uniform(0.1, 1.0, n_pts))
            raw_precs   = np.clip(np.random.uniform(0.3, 1.0, n_pts) - np.linspace(0, 0.3, n_pts), 0, 1)
            per_query_curves[qid] = (list(raw_recalls), list(raw_precs))
            ap_scores_test[qid] = float(np.mean(raw_precs))

        # identify best/worst/extra by AP
        sorted_by_ap = sorted(ap_scores_test.items(), key=lambda x: x[1])
        highlight_ids_test = {
            "worst": sorted_by_ap[0][0],
            "extra": sorted_by_ap[len(sorted_by_ap) // 2][0],
            "best":  sorted_by_ap[-1][0],
        }

        p5 = os.path.join(tmpdir, "individual_pr_curves.png")
        fig5 = plot_individual_pr_curves(
            per_query_curves,
            ap_scores_test,
            highlight_ids_test,
            strategy_name="BM25",
            save_path=p5,
        )
        plt.close(fig5)
        assert os.path.exists(p5), f"File not created: {p5}"
        print(f"  plot_individual_pr_curves: OK ({p5})")

    # temp dir auto-deleted
    print("  Temp files cleaned up.")

    print("\n" + "=" * 60)
    print("plots.py — all self-tests passed (no crash, files created)")
    print("=" * 60)

