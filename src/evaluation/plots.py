"""
Visualisation functions for Phase 1 retrieval evaluation.
!!! AI used for better formats, consistent colors, and value labels on bars.
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



# --- Plot: NDCG scale sensitivity (§3.1.1) ------------------------------------

def plot_ndcg_scale_sensitivity(
    results: dict[str, dict[str, float]],
    scale_labels: list[str] | None = None,
    title: str = "NDCG@100 Stability Across Relevance Scales",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Grouped bar chart showing NDCG@100 for each strategy across multiple
    qrel relevance scales (e.g., 0-2, 0-5, 0-7).

    Parameters
    ----------
    results      : {strategy_name: {scale_label: ndcg_value}}
    scale_labels : ordered list of scale labels to display; if None, inferred from results
    title        : plot title
    save_path    : optional path to save the figure

    Returns
    -------
    matplotlib Figure
    """
    strategies = list(results.keys())
    if scale_labels is None:
        scale_labels = list(next(iter(results.values())).keys())

    n_strategies = len(strategies)
    n_scales     = len(scale_labels)
    x = np.arange(n_strategies)
    bar_w = 0.8 / n_scales

    scale_colors = ["#90B8E0", "#1F6EBD", "#0A2A5A"][:n_scales]

    fig, ax = plt.subplots(figsize=(max(8, n_strategies * 1.4), 5))

    for i, (scale_lbl, color) in enumerate(zip(scale_labels, scale_colors)):
        vals = [results[s].get(scale_lbl, 0) for s in strategies]
        offset = (i - n_scales / 2 + 0.5) * bar_w
        bars = ax.bar(x + offset, vals, bar_w, label=scale_lbl, color=color,
                      edgecolor="white", linewidth=0.6)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{bar.get_height():.4f}",
                    ha="center", va="bottom", fontsize=7.5, color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=10, rotation=15, ha="right")
    ax.set_ylabel("NDCG@100", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.08)
    ax.legend(title="Relevance scale", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plots] Saved scale sensitivity -> {save_path}")
    return fig


# --- Plot: Baseline vs Tuned — all metrics (§4.3) ----------------------------

def plot_baseline_vs_tuned(
    baseline_results: dict[str, dict],
    tuned_results: dict[str, dict],
    base_strats: list[str]  | None = None,
    tuned_strats: list[str] | None = None,
    display_names: list[str]| None = None,
    metrics: list[str] | None = None,
    title: str = "Baseline vs Tuned — All Metrics on Test Set",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Grouped bar chart (one subplot per metric) comparing baseline vs tuned
    performance for each strategy.

    Parameters
    ----------
    baseline_results : {strategy_name: {metric: float}}
    tuned_results    : {strategy_name: {metric: float}}
    base_strats      : ordered list of keys into baseline_results
    tuned_strats     : ordered list of keys into tuned_results (same order as base_strats)
    display_names    : x-axis labels (same length as base_strats)
    metrics          : list of metric names to plot (default: MAP, MRR, P@10, R@100, NDCG@100)
    title            : suptitle
    save_path        : optional save path

    Returns
    -------
    matplotlib Figure
    """
    if metrics is None:
        metrics = ["MAP", "MRR", "P@10", "R@100", "NDCG@100"]
    metric_labels = [m.replace("@", "\n@") for m in metrics]

    if base_strats is None:
        base_strats = list(baseline_results.keys())
    if tuned_strats is None:
        tuned_strats = list(tuned_results.keys())
    if display_names is None:
        display_names = base_strats

    n = len(display_names)
    colors_base  = ["#90B8E0", "#F5A97A", "#A3D9A5", "#D9A5D9", "#F5D57A"][:n]
    colors_tuned = ["#1F6EBD", "#D05A14", "#2E8B57", "#8B2FC9", "#C9A800"][:n]

    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 5), sharey=False)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    x    = np.arange(n)
    bw   = 0.35

    for ax, metric, mlabel in zip(axes, metrics, metric_labels):
        bvals = [baseline_results.get(b, {}).get(metric, 0) for b in base_strats]
        tvals = [tuned_results.get(t, {}).get(metric, 0)    for t in tuned_strats]

        bars_b = ax.bar(x - bw / 2, bvals, bw, color=colors_base,
                        edgecolor="white", linewidth=0.5)
        bars_t = ax.bar(x + bw / 2, tvals, bw, color=colors_tuned,
                        edgecolor="white", linewidth=0.5)

        for bar in bars_b:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom",
                    fontsize=6.5, color="#555")
        for bar in bars_t:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom",
                    fontsize=6.5, fontweight="bold", color="#111")

        ax.set_title(mlabel, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(display_names, fontsize=8, rotation=20, ha="right")
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    import matplotlib.patches as mpatches
    legend_patches = [
        mpatches.Patch(color="#888888", label="Baseline (default params)"),
        mpatches.Patch(color="#222222", label="Tuned (locked §3 params)"),
    ]
    fig.legend(handles=legend_patches, loc="upper right", ncol=2,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plots] Saved baseline_vs_tuned -> {save_path}")
    return fig


# --- Plot: Tuning gain — Δ NDCG@100 and Δ MAP (§4.3) -------------------------

def plot_tuning_gain(
    baseline_results: dict[str, dict],
    tuned_results: dict[str, dict],
    pairs: list[tuple[str, str]] | None = None,
    comp_labels: list[str] | None = None,
    title: str = "Tuning Gain on Test Set (Tuned − Baseline)",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Side-by-side bar charts showing Δ NDCG@100 and Δ MAP per strategy
    (tuned score minus baseline score).

    Parameters
    ----------
    baseline_results : {strategy_name: {metric: float}}
    tuned_results    : {strategy_name: {metric: float}}
    pairs            : [(baseline_key, tuned_key), ...] — matched strategy pairs
    comp_labels      : display labels for x-axis (same length as pairs)
    title            : suptitle
    save_path        : optional save path

    Returns
    -------
    matplotlib Figure
    """
    if pairs is None:
        pairs = [
            ("BM25",    "BM25 (tuned)"),
            ("LM-Dir",  "LM-Dir (mu=75)"),
            ("KNN",     "KNN (MedCPT)"),
            ("RRF",     "RRF (tuned)"),
        ]
    if comp_labels is None:
        comp_labels = [b for b, _ in pairs]

    delta_ndcg, delta_map = [], []
    for bkey, tkey in pairs:
        b = baseline_results.get(bkey, {})
        t = tuned_results.get(tkey, {})
        delta_ndcg.append(t.get("NDCG@100", 0) - b.get("NDCG@100", 0))
        delta_map.append( t.get("MAP",      0) - b.get("MAP",      0))

    fig, (ax_ndcg, ax_map) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    def _gain_bar(ax, deltas, subtitle,
                  color_pos: str = "#2E8B57", color_neg: str = "#C0392B") -> None:
        xpos   = np.arange(len(comp_labels))
        colors = [color_pos if d >= 0 else color_neg for d in deltas]
        bars   = ax.bar(xpos, deltas, color=colors,
                        edgecolor="white", linewidth=0.7, width=0.55)
        ax.axhline(0, color="black", linewidth=0.8)
        for bar, d in zip(bars, deltas):
            yoff = 0.0015 if d >= 0 else -0.004
            va   = "bottom" if d >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + yoff,
                    f"{d:+.4f}", ha="center", va=va,
                    fontsize=10, fontweight="bold", color="#111")
        ax.set_xticks(xpos)
        ax.set_xticklabels(comp_labels, fontsize=11)
        ax.set_title(subtitle, fontsize=11, fontweight="bold")
        ax.set_ylabel("Δ metric (tuned − baseline)", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    _gain_bar(ax_ndcg, delta_ndcg, "Δ NDCG@100")
    _gain_bar(ax_map,  delta_map,  "Δ MAP")

    ax_ndcg.text(
        0.02, 0.97,
        "Green = tuning helped on test set\nRed = tuning had no effect or hurt",
        transform=ax_ndcg.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="#ccc", alpha=0.8),
    )

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plots] Saved tuning_gain -> {save_path}")

    # print summary to stdout
    for lbl, dn, dm in zip(comp_labels, delta_ndcg, delta_map):
        direction = ("improved" if dn > 0.001
                     else "negligible" if abs(dn) <= 0.001
                     else "slightly hurt")
        print(f"  {lbl:<12} ΔNDCG@100={dn:+.4f}  ΔMAP={dm:+.4f}  → {direction}")

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

