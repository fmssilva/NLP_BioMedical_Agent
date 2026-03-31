"""
src/tuning/tuning_plots.py

Shared plotting helpers for all hyperparameter sweep notebooks.
Each function takes data, produces a figure, optionally saves to disk, and returns the figure.
No global state, no side effects beyond the optional save.

Three helpers:
  - plot_sweep_bar   : bar+errorbar for any 1D parameter sweep (MAP vs one param)
  - plot_heatmap_2d  : colour heatmap for 2D grid search (BM25 k1 x b)
  - plot_encoder_bars: grouped bar chart comparing MAP/MRR/P@10 across encoders
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from typing import Optional


# consistent colour palette across all tuning plots
BASELINE_COLOR = "tomato"
BEST_COLOR     = "steelblue"
OTHER_COLOR    = "steelblue"

# encoder-specific colours
ENCODER_COLORS = {
    "msmarco": "tomato",    # baseline — red so it stands out
    "medcpt":  "steelblue",
    "multi-qa": "seagreen",
}


# ── 1D sweep bar chart ──────────────────────────────────────────────────────

def plot_sweep_bar(
    param_values:  list,          # x-axis tick values (e.g. [50, 75, 100, 200, 500, ...])
    mean_maps:     list[float],   # mean MAP per param value
    std_maps:      list[float],   # std MAP per param value (for error bars)
    baseline_val,                 # which param value is the baseline (highlighted red)
    param_label:   str,           # x-axis label, e.g. "mu" or "lambda"
    title:         str,
    save_path:     Optional[Path] = None,
) -> plt.Figure:
    """
    Bar chart with error bars for a 1D parameter sweep.

    Baseline bar is red, all others blue. The best MAP bar gets a '*' label.
    Error bars show the cross-validation std across folds.
    """
    fig, ax = plt.subplots(figsize=(max(6, len(param_values) * 1.1), 4))

    x_labels = [str(v) for v in param_values]
    best_idx  = int(np.argmax(mean_maps))

    colors = [
        BASELINE_COLOR if v == baseline_val else BEST_COLOR
        for v in param_values
    ]

    bars = ax.bar(
        x_labels, mean_maps, yerr=std_maps,
        color=colors, capsize=5, edgecolor="white", linewidth=0.5,
        error_kw={"elinewidth": 1.2, "ecolor": "gray"},
    )

    # baseline reference line
    base_idx = param_values.index(baseline_val) if baseline_val in param_values else None
    if base_idx is not None:
        ax.axhline(mean_maps[base_idx], color=BASELINE_COLOR, linestyle="--",
                   alpha=0.5, label=f"Baseline ({param_label}={baseline_val})")

    # value labels above bars
    for i, (bar, v) in enumerate(zip(bars, mean_maps)):
        label = f"{v:.4f}" + (" *" if i == best_idx else "")
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(std_maps) * 0.3,
                label, ha="center", va="bottom", fontsize=8,
                fontweight="bold" if i == best_idx else "normal")

    ax.set_xlabel(param_label)
    ax.set_ylabel("Mean MAP (5-fold CV)")
    ax.set_title(title)
    ax.legend(fontsize=8)

    # y-axis range: give some breathing room
    ymin = max(0, min(mean_maps) - max(std_maps) * 0.8)
    ymax = max(mean_maps) + max(std_maps) * 1.5
    ax.set_ylim(ymin, ymax)

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")

    return fig


# ── 2D heatmap (BM25 k1/b or any 2-param grid) ─────────────────────────────

def plot_heatmap_2d(
    row_values:    list[float],   # y-axis param (e.g. k1 values)
    col_values:    list[float],   # x-axis param (e.g. b values)
    map_matrix:    np.ndarray,    # shape (len(row_values), len(col_values)) — MAP values
    row_label:     str,           # y-axis label
    col_label:     str,           # x-axis label
    baseline_row:  float,         # row value of the baseline config (drawn as red border)
    baseline_col:  float,         # col value of the baseline config
    title:         str,
    save_path:     Optional[Path] = None,
) -> plt.Figure:
    """
    Colour heatmap for a 2D grid search result.

    Best cell gets a '*', baseline cell gets a red border. Values annotated inside cells.
    """
    fig, ax = plt.subplots(figsize=(max(6, len(col_values) * 1.5), max(4, len(row_values) * 1.2)))

    im = ax.imshow(
        map_matrix, cmap="YlGn", aspect="auto",
        vmin=map_matrix.min() - 0.005, vmax=map_matrix.max() + 0.005,
    )
    plt.colorbar(im, ax=ax, label="Mean MAP (5-fold CV)")

    # axis ticks
    ax.set_xticks(range(len(col_values)))
    ax.set_xticklabels([f"{col_label}={c}" for c in col_values])
    ax.set_yticks(range(len(row_values)))
    ax.set_yticklabels([f"{row_label}={r}" for r in row_values])

    # find best cell
    best_i, best_j = np.unravel_index(np.argmax(map_matrix), map_matrix.shape)

    # annotate cells with values
    for i in range(len(row_values)):
        for j in range(len(col_values)):
            v    = map_matrix[i, j]
            star = "*" if (i == best_i and j == best_j) else ""
            ax.text(j, i, f"{v:.4f}{star}", ha="center", va="center", fontsize=8)

    # red border around baseline cell
    try:
        bi = row_values.index(baseline_row)
        bj = col_values.index(baseline_col)
        ax.add_patch(plt.Rectangle(
            (bj - 0.5, bi - 0.5), 1, 1,
            fill=False, edgecolor="red", lw=2, label="Baseline",
        ))
        ax.legend(loc="upper right", fontsize=8)
    except ValueError:
        pass   # baseline not in grid — skip border

    ax.set_title(title)
    ax.set_xlabel(col_label)
    ax.set_ylabel(row_label)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")

    return fig


# ── Encoder comparison grouped bar chart ────────────────────────────────────

def plot_encoder_bars(
    encoder_keys:    list[str],          # e.g. ["msmarco", "medcpt", "multi-qa"]
    encoder_labels:  list[str],          # display names (can be long)
    map_values:      list[float],
    mrr_values:      list[float],
    p10_values:      list[float],
    baseline_key:    str = "msmarco",    # which encoder is the baseline
    title:           str = "Dense Encoder Comparison",
    save_path:       Optional[Path] = None,
) -> plt.Figure:
    """
    Three-panel grouped bar chart: MAP | MRR | P@10 per encoder.
    Baseline encoder highlighted in red; others in blue/green.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    metrics   = [map_values, mrr_values, p10_values]
    titles_   = ["MAP", "MRR", "P@10"]

    # shorten labels for x-axis (avoid overlap)
    short_labels = [_shorten_label(lbl) for lbl in encoder_labels]

    colors = [
        ENCODER_COLORS.get(k, OTHER_COLOR) for k in encoder_keys
    ]

    for ax, vals, mtitle in zip(axes, metrics, titles_):
        bars = ax.bar(short_labels, vals, color=colors, edgecolor="white")
        ax.set_title(mtitle, fontsize=11)
        ymin = max(0, min(vals) - 0.06)
        ymax = max(vals) + 0.05
        ax.set_ylim(ymin, ymax)
        ax.tick_params(axis="x", rotation=12, labelsize=8)

        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.006,
                f"{v:.4f}", ha="center", va="bottom", fontsize=8,
            )

    # shared legend: baseline vs others
    legend_patches = [
        mpatches.Patch(color=BASELINE_COLOR, label="Baseline (msmarco)"),
        mpatches.Patch(color=BEST_COLOR,     label="Candidate"),
    ]
    fig.legend(handles=legend_patches, loc="upper right", fontsize=8)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0, 0, 0.92, 1])

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")

    return fig


def _shorten_label(label: str, max_len: int = 22) -> str:
    """Trim a long encoder name for x-axis readability."""
    # take the last part after "/" if it's a HuggingFace path
    if "/" in label:
        label = label.split("/")[-1]
    return label[:max_len]


# ── PR curve interpretation helper ──────────────────────────────────────────

def plot_pr_interpretation(
    real_strategy_curves: dict,   # {name: (recall_array, precision_array)} — our actual results
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Two-row figure:
      Row 1 (3 panels): dummy 'perfect', 'random', and 'typical good' PR curves for interpretation
      Row 2 (1 wide panel): our actual mean PR curves for all strategies

    This teaches the reader how to read PR plots before they see the real results.
    """
    fig = plt.figure(figsize=(14, 9))

    # ── Row 1: three reference shapes ───────────────────────────────────────
    recall = np.linspace(0, 1, 50)

    # perfect retriever: precision stays 1.0 all the way to recall=1
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(recall, np.ones_like(recall), "b-o", markersize=3, label="Perfect (AUC=1.0)")
    ax1.fill_between(recall, np.ones_like(recall), alpha=0.15)
    ax1.set_title("Perfect retriever", fontsize=10)
    ax1.set_ylim(-0.05, 1.1); ax1.set_xlim(-0.02, 1.02)
    ax1.set_xlabel("Recall"); ax1.set_ylabel("Precision")
    ax1.legend(fontsize=8)

    # random baseline: precision ≈ fraction of relevant docs in corpus (~0.01 for this task)
    random_prec = 0.011 * np.ones_like(recall)   # 46 relevant / 4194 corpus ≈ 1.1%
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(recall, random_prec, "r--o", markersize=3, label="Random (chance)")
    ax2.fill_between(recall, random_prec, alpha=0.12, color="red")
    ax2.set_title("Random retriever (chance)", fontsize=10)
    ax2.set_ylim(-0.05, 1.1); ax2.set_xlim(-0.02, 1.02)
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax2.legend(fontsize=8)

    # typical good IR: starts high, drops gracefully (concave curve)
    good_prec = 1.0 / (1 + 2.5 * recall)   # hyperbola-ish, concave
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(recall, good_prec, "g-o", markersize=3, label="Good retriever (~0.55 MAP)")
    ax3.fill_between(recall, good_prec, alpha=0.15, color="green")
    ax3.set_title("Typical good retriever", fontsize=10)
    ax3.set_ylim(-0.05, 1.1); ax3.set_xlim(-0.02, 1.02)
    ax3.set_xlabel("Recall"); ax3.set_ylabel("Precision")
    ax3.legend(fontsize=8)

    # ── Row 2: our actual PR curves ──────────────────────────────────────────
    ax_real = fig.add_subplot(2, 1, 2)

    STRATEGY_COLORS = {
        "BM25":   "steelblue",
        "LM-JM":  "tomato",
        "LM-Dir": "seagreen",
        "KNN":    "darkorange",
        "RRF":    "purple",
    }
    for name, (rl, mp) in real_strategy_curves.items():
        color = STRATEGY_COLORS.get(name, "gray")
        ax_real.plot(rl, mp, "o-", color=color, markersize=3, label=name)
        ax_real.fill_between(rl, mp, alpha=0.07, color=color)

    ax_real.set_title("Our actual results — Phase 1 test set (33 queries)", fontsize=10)
    ax_real.set_xlabel("Recall")
    ax_real.set_ylabel("Precision")
    ax_real.legend(loc="upper right", fontsize=8)
    ax_real.set_ylim(-0.02, 1.05)
    ax_real.set_xlim(-0.02, 1.02)
    ax_real.grid(alpha=0.3)

    # annotation box explaining what to read
    note = (
        "How to read: area under the curve = MAP. "
        "Curves that stay high across all recall levels are best.\n"
        "A steep early drop means the retriever finds relevant docs early but misses many at high recall."
    )
    fig.text(0.01, 0.48, note, fontsize=8, color="dimgray", style="italic",
             wrap=True, va="top")

    fig.suptitle("How to interpret Precision-Recall curves", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")

    return fig


# ── self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile, os

    print("Testing plot_sweep_bar ...")
    fig = plot_sweep_bar(
        param_values=[50, 75, 100, 200, 500, 1000, 2000],
        mean_maps=[0.57, 0.565, 0.553, 0.546, 0.536, 0.529, 0.520],
        std_maps=[0.09] * 7,
        baseline_val=2000,
        param_label="mu",
        title="LM-Dir mu sweep (self-test)",
    )
    assert fig is not None
    plt.close(fig)
    print("  OK")

    print("Testing plot_heatmap_2d ...")
    k1s = [0.5, 0.8, 1.0, 1.2, 1.5]
    bs  = [0.25, 0.5, 0.75, 1.0]
    mat = np.random.uniform(0.54, 0.59, (len(k1s), len(bs)))
    fig = plot_heatmap_2d(k1s, bs, mat, "k1", "b", 1.2, 0.75, "BM25 sweep (self-test)")
    assert fig is not None
    plt.close(fig)
    print("  OK")

    print("Testing plot_encoder_bars ...")
    fig = plot_encoder_bars(
        encoder_keys=["msmarco", "medcpt", "multi-qa"],
        encoder_labels=["msmarco-distilbert-base-v2", "MedCPT (asymmetric)", "multi-qa-mpnet-base-cos-v1"],
        map_values=[0.4268, 0.6095, 0.5273],
        mrr_values=[0.7979, 0.8568, 0.8307],
        p10_values=[0.6344, 0.7250, 0.6937],
        title="Encoder comparison (self-test)",
    )
    assert fig is not None
    plt.close(fig)
    print("  OK")

    print("Testing plot_pr_interpretation ...")
    dummy_curves = {
        "BM25":  (np.linspace(0, 1, 11), 1 / (1 + 2.2 * np.linspace(0, 1, 11))),
        "KNN":   (np.linspace(0, 1, 11), 1 / (1 + 3.5 * np.linspace(0, 1, 11))),
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "pr_interp_test.png"
        fig = plot_pr_interpretation(dummy_curves, save_path=path)
        assert path.exists(), "PNG not saved"
    plt.close(fig)
    print("  OK")

    print("All tuning_plots self-tests passed.")
