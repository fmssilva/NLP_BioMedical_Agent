"""
!!!! AI USED FOR PLOT FORMATTING
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from typing import Optional


# consistent colour palette across all tuning plots
# baseline = muted gold/sand (visible but not aggressive)
# best     = seagreen (clearly "winner")
# other    = steelblue (neutral)
BASELINE_COLOR = "#C9A84C"   # muted gold
BEST_COLOR     = "#2E8B57"   # seagreen
OTHER_COLOR    = "#4A90C4"   # steel blue

# encoder-specific colours (key → bar colour)
# msmarco is the baseline → gold; best winner → seagreen; others → steel blue
ENCODER_COLORS = {
    "msmarco":  BASELINE_COLOR,
    "medcpt":   BEST_COLOR,
    "multi-qa": OTHER_COLOR,
}


# ── 1D sweep bar chart ──────────────────────────────────────────────────────

def plot_sweep_bar(
    param_values:  list,          # x-axis tick values (e.g. [50, 75, 100, 200, 500, ...])
    mean_maps:     list[float],   # mean MAP per param value
    std_maps:      list[float],   # std MAP per param value (for error bars)
    baseline_val,                 # which param value is the baseline (highlighted gold)
    param_label:   str,           # x-axis label, e.g. "mu" or "lambda"
    title:         str,
    save_path:     Optional[Path] = None,
) -> plt.Figure:
    """
    Bar chart with error bars for a 1D parameter sweep.

    Baseline bar is muted gold, best MAP bar is seagreen, all others are steel blue.
    Error bars show the cross-validation std across folds.
    """
    fig, ax = plt.subplots(figsize=(max(6, len(param_values) * 1.1), 4))

    x_labels = [str(v) for v in param_values]
    best_idx  = int(np.argmax(mean_maps))
    base_idx  = param_values.index(baseline_val) if baseline_val in param_values else None

    colors = []
    for i, v in enumerate(param_values):
        if i == best_idx:
            colors.append(BEST_COLOR)
        elif v == baseline_val:
            colors.append(BASELINE_COLOR)
        else:
            colors.append(OTHER_COLOR)

    bars = ax.bar(
        x_labels, mean_maps, yerr=std_maps,
        color=colors, capsize=5, edgecolor="white", linewidth=0.5,
        error_kw={"elinewidth": 1.2, "ecolor": "gray"},
    )

    # baseline reference line
    if base_idx is not None:
        ax.axhline(mean_maps[base_idx], color=BASELINE_COLOR, linestyle="--",
                   linewidth=1.5, alpha=0.7, label=f"Baseline ({param_label}={baseline_val})")

    # best reference line
    ax.axhline(mean_maps[best_idx], color=BEST_COLOR, linestyle=":",
               linewidth=1.5, alpha=0.7, label=f"Best ({param_label}={param_values[best_idx]})")

    # value labels above bars
    for i, (bar, v) in enumerate(zip(bars, mean_maps)):
        is_best = (i == best_idx)
        label = f"{v:.4f}" + (" ★" if is_best else "")
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(std_maps) * 0.3,
                label, ha="center", va="bottom", fontsize=8,
                fontweight="bold" if is_best else "normal",
                color=BEST_COLOR if is_best else "black")

    ax.set_xlabel(param_label)
    ax.set_ylabel("Mean NDCG@100 (5-fold CV)")
    ax.set_title(title)
    ax.legend(fontsize=8)

    # y-axis range: give some breathing room
    ymin = max(0, min(mean_maps) - max(std_maps) * 0.8)
    ymax = max(mean_maps) + max(std_maps) * 1.8
    ax.set_ylim(ymin, ymax)

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")

    plt.close(fig)
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
            star = " ★" if (i == best_i and j == best_j) else ""
            ax.text(j, i, f"{v:.4f}{star}", ha="center", va="center", fontsize=8,
                    fontweight="bold" if (i == best_i and j == best_j) else "normal",
                    color="black")

    # green border around best cell
    ax.add_patch(plt.Rectangle(
        (best_j - 0.5, best_i - 0.5), 1, 1,
        fill=False, edgecolor=BEST_COLOR, lw=2.5, label="Best",
    ))

    # gold dashed border around baseline cell
    try:
        bi = row_values.index(baseline_row)
        bj = col_values.index(baseline_col)
        ax.add_patch(plt.Rectangle(
            (bj - 0.5, bi - 0.5), 1, 1,
            fill=False, edgecolor=BASELINE_COLOR, lw=2, linestyle="--", label="Baseline",
        ))
        ax.legend(loc="upper right", fontsize=8)
    except ValueError:
        ax.legend(loc="upper right", fontsize=8)

    ax.set_title(title)
    ax.set_xlabel(col_label)
    ax.set_ylabel(row_label)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")

    plt.close(fig)
    return fig


# ── Encoder comparison grouped bar chart ────────────────────────────────────

def plot_encoder_bars(
    encoder_keys:    list[str],          # e.g. ["msmarco", "medcpt", "multi-qa"]
    encoder_labels:  list[str],          # display names (can be long)
    ndcg_values:     list[float],
    map_values:      list[float],
    mrr_values:      list[float],
    p10_values:      list[float],
    baseline_key:    str = "msmarco",    # which encoder is the baseline
    title:           str = "Dense Encoder Comparison",
    save_path:       Optional[Path] = None,
) -> plt.Figure:
    """
    Four-panel grouped bar chart: NDCG@100 | MAP | MRR | P@10 per encoder.
    Baseline encoder highlighted in gold; best in seagreen; others in steel blue.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    metrics   = [ndcg_values, map_values, mrr_values, p10_values]
    titles_   = ["NDCG@100 ★", "MAP", "MRR", "P@10"]

    # shorten labels for x-axis (avoid overlap)
    short_labels = [_shorten_label(lbl) for lbl in encoder_labels]

    colors = [
        ENCODER_COLORS.get(k, OTHER_COLOR) for k in encoder_keys
    ]

    for ax, vals, mtitle in zip(axes, metrics, titles_):
        best_i = int(np.argmax(vals))
        bars = ax.bar(short_labels, vals, color=colors, edgecolor="white")
        ax.set_title(mtitle, fontsize=11)
        ymin = max(0, min(vals) - 0.06)
        ymax = max(vals) + 0.07
        ax.set_ylim(ymin, ymax)
        ax.tick_params(axis="x", rotation=0, labelsize=9)

        for i, (bar, v) in enumerate(zip(bars, vals)):
            is_best = (i == best_i)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.006,
                f"{v:.4f}" + (" ★" if is_best else ""),
                ha="center", va="bottom", fontsize=8,
                fontweight="bold" if is_best else "normal",
                color=BEST_COLOR if is_best else "black",
            )

    # shared legend: baseline vs best vs others
    legend_patches = [
        mpatches.Patch(color=BASELINE_COLOR, label="Baseline (msmarco)"),
        mpatches.Patch(color=BEST_COLOR,     label="Best encoder ★"),
        mpatches.Patch(color=OTHER_COLOR,    label="Other candidates"),
    ]
    fig.legend(handles=legend_patches, loc="upper right", fontsize=8)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0, 0, 0.92, 1])

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")

    plt.close(fig)
    return fig



# ── Field ablation bar chart ─────────────────────────────────────────────────

def plot_field_ablation(
    ablation_results: dict,           # {field_name: {"NDCG@100": float, "MAP": float, "MRR": float, "P@10": float}}
    primary_metric:   str  = "NDCG@100",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Grouped bar chart for query field ablation: NDCG@100 / MAP / MRR / P@10 per field.

    Best field (by primary_metric) → seagreen bars.
    All others → steel blue bars.
    Supports 3–6 fields; labels are wrapped automatically.
    """
    fields   = list(ablation_results.keys())
    metrics  = ["NDCG@100", "MAP", "MRR", "P@10"]
    n_fields = len(fields)
    n_metrics = len(metrics)

    # determine winner
    best_field = max(fields, key=lambda f: ablation_results[f].get(primary_metric, 0.0))

    # field display labels — wrap long names so they fit on the x-axis
    _label_map = {
        "topic":           "topic",
        "question":        "question",
        "narrative":       "narrative",
        "topic+question":  "topic\n+question",
        "topic+narrative": "topic\n+narrative",
        "concatenated":    "concatenated\n(all three)",
    }
    x_labels = [_label_map.get(f, f) for f in fields]

    # scale figure width with number of fields
    fig_w = max(14, n_fields * 2.5)
    fig, axes = plt.subplots(1, n_metrics, figsize=(fig_w, 4.8), sharey=False)

    metric_label_map = {
        "NDCG@100": "NDCG@100 ★ primary",
        "MAP":     "MAP",
        "MRR":     "MRR",
        "P@10":    "P@10",
    }

    for ax, metric in zip(axes, metrics):
        vals = [ablation_results[f].get(metric, 0.0) for f in fields]
        best_i = fields.index(best_field)

        bar_colors = [
            BEST_COLOR if f == best_field else OTHER_COLOR
            for f in fields
        ]

        bars = ax.bar(x_labels, vals, color=bar_colors, edgecolor="white", width=0.55)

        ymin = max(0.0, min(vals) - 0.04)
        ymax = max(vals) + 0.06
        ax.set_ylim(ymin, ymax)
        ax.set_title(metric_label_map.get(metric, metric), fontsize=10,
                     fontweight="bold" if metric == primary_metric else "normal",
                     color=BEST_COLOR if metric == primary_metric else "black")
        ax.tick_params(axis="x", labelsize=7.5, rotation=45)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")

        for i, (bar, v) in enumerate(zip(bars, vals)):
            is_best = (i == best_i) and (metric == primary_metric)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.004,
                f"{v:.4f}" + (" ★" if is_best else ""),
                ha="center", va="bottom", fontsize=7.5,
                fontweight="bold" if (i == best_i) else "normal",
                color=BEST_COLOR if (i == best_i) else "black",
            )

    # legend
    legend_patches = [
        mpatches.Patch(color=BEST_COLOR,  label=f"Best field ({best_field}) ★"),
        mpatches.Patch(color=OTHER_COLOR, label="Other fields"),
    ]
    fig.legend(handles=legend_patches, loc="upper right", fontsize=9)

    fig.suptitle(
        f"Query Field Ablation (BM25, {n_fields} fields, 32 train topics)\n"
        f"Primary criterion: {primary_metric} → winner: '{best_field}'",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 0.87, 0.93])

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plots] Saved field ablation -> {save_path}")

    plt.close(fig)
    return fig


# ── High-level SweepResult wrappers ─────────────────────────────────────────
# Each accepts a SweepResult object (from src.tuning.sweeper) and delegates
# to the low-level helpers above.  These are the functions called from the notebook.

def plot_lmdir_sweep(result, save_path: Optional[Path] = None) -> plt.Figure:
    """
    Bar chart for LM-Dirichlet mu sweep. Accepts a SweepResult (kind="lmdir").
    """
    rows = sorted(result.rows, key=lambda r: r["mu"])
    return plot_sweep_bar(
        param_values  = [r["mu"] for r in rows],
        mean_maps     = [r["mean_ndcg"] for r in rows],
        std_maps      = [r["std_ndcg"] for r in rows],
        baseline_val  = result.baseline_id,
        param_label   = "μ",
        title         = "LM-Dirichlet μ — NDCG@100 (5-fold CV on train)",
        save_path     = save_path,
    )


def plot_lmjm_sweep(result, save_path: Optional[Path] = None) -> plt.Figure:
    """
    Bar chart for LM Jelinek-Mercer lambda sweep. Accepts a SweepResult (kind="lmjm").
    """
    rows = sorted(result.rows, key=lambda r: r["lambda"])
    return plot_sweep_bar(
        param_values  = [r["lambda"] for r in rows],
        mean_maps     = [r["mean_ndcg"] for r in rows],
        std_maps      = [r["std_ndcg"] for r in rows],
        baseline_val  = result.baseline_id,
        param_label   = "λ",
        title         = "LM Jelinek-Mercer λ — NDCG@100 (5-fold CV on train)",
        save_path     = save_path,
    )


def plot_bm25_sweep(result, save_path: Optional[Path] = None) -> plt.Figure:
    """
    Heatmap for BM25 (k1, b) grid sweep. Accepts a SweepResult (kind="bm25").
    """
    k1_vals = sorted(set(r["k1"] for r in result.rows))
    b_vals  = sorted(set(r["b"]  for r in result.rows))

    # build map matrix (rows=k1, cols=b)
    lookup = {(r["k1"], r["b"]): r["mean_ndcg"] for r in result.rows}
    matrix = np.array(
        [[lookup.get((k1, b), 0.0) for b in b_vals] for k1 in k1_vals]
    )

    baseline_k1, baseline_b = result.baseline_id
    return plot_heatmap_2d(
        row_values   = k1_vals,
        col_values   = b_vals,
        map_matrix   = matrix,
        row_label    = "k1",
        col_label    = "b",
        baseline_row = baseline_k1,
        baseline_col = baseline_b,
        title        = "BM25 (k1, b) — NDCG@100 (5-fold CV on train)",
        save_path    = save_path,
    )


def plot_encoder_sweep(result, save_path: Optional[Path] = None) -> plt.Figure:
    """
    Grouped bar chart comparing encoders. Accepts a SweepResult (kind="encoder").
    """
    rows = result.rows   # already sorted by ndcg desc
    return plot_encoder_bars(
        encoder_keys   = [r["alias"] for r in rows],
        encoder_labels = [r["alias"] for r in rows],
        ndcg_values    = [r["ndcg"]  for r in rows],
        map_values     = [r["map"]   for r in rows],
        mrr_values     = [r["mrr"]   for r in rows],
        p10_values     = [r["p10"]   for r in rows],
        baseline_key   = result.baseline_id,
        title          = "Dense Encoder Comparison — NDCG@100 / MAP / MRR / P@10 (train set)",
        save_path      = save_path,
    )


def _shorten_label(label: str, max_len: int = 22) -> str:
    """Trim a long encoder name for x-axis readability."""
    # take the last part after "/" if it's a HuggingFace path
    if "/" in label:
        label = label.split("/")[-1]
    return label[:max_len]


# ── RRF pair comparison bar chart ────────────────────────────────────────────

_SOLO_BAR_COLOR = "#e07b54"   # orange — all solo-model bars share this colour


def plot_rrf_bars(
    labels:       list[str],   # one label per (pair, k) combination
    ndcg_values:  list[float],
    map_values:   list[float],
    mrr_values:   list[float],
    p10_values:   list[float],
    best_rrf_idx: int,         # index of the best RRF config in the (already merged+sorted) list
    is_solo:      list[bool],  # True for solo-model bars, False for RRF-pair bars
    title:        str = "RRF Pair Grid Search",
    save_path:    Optional[Path] = None,
) -> plt.Figure:
    """
    2×2 bar chart (NDCG@100 | MAP | MRR | P@10) for RRF pair × k combinations,
    with solo-model reference bars injected and sorted in the same chart.

    Colour coding:
      seagreen  — best RRF config (★)
      steelblue — other RRF configs
      orange    — solo-model best scores (labelled "… best solo")
    X-labels rotated 45° so they fit.
    """
    n = len(labels)
    bar_colors = []
    for i in range(n):
        if is_solo[i]:
            bar_colors.append(_SOLO_BAR_COLOR)
        elif i == best_rrf_idx:
            bar_colors.append(BEST_COLOR)
        else:
            bar_colors.append(OTHER_COLOR)

    fig, axes_2d = plt.subplots(2, 2, figsize=(max(12, n * 1.0), 9.6))
    axes = axes_2d.flatten()

    metrics_data = [ndcg_values, map_values, mrr_values, p10_values]
    titles_      = ["NDCG@100 ★", "MAP",      "MRR",      "P@10"]

    for ax, vals, mtitle in zip(axes, metrics_data, titles_):
        # best overall bar (may be a solo)
        b_idx = int(np.argmax(vals))
        # best RRF-only bar
        rrf_best_v = max(
            (v for i, v in enumerate(vals) if not is_solo[i]), default=None
        )
        bars = ax.bar(labels, vals, color=bar_colors, edgecolor="white")
        ax.set_title(mtitle, fontsize=11,
                     fontweight="bold" if mtitle == "NDCG@100 ★" else "normal",
                     color=BEST_COLOR if mtitle == "NDCG@100 ★" else "black")
        ymin = max(0, min(vals) - 0.04)
        ymax = max(vals) + 0.07
        ax.set_ylim(ymin, ymax)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")

        # value annotations
        for i, (bar, v) in enumerate(zip(bars, vals)):
            is_overall_best = (i == b_idx)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.006,
                f"{v:.4f}" + (" ★" if is_overall_best else ""),
                ha="center", va="bottom", fontsize=7.5,
                fontweight="bold" if is_overall_best else "normal",
                color=BEST_COLOR if (not is_solo[i] and i == best_rrf_idx)
                      else (_SOLO_BAR_COLOR if is_solo[i] else "black"),
            )

    legend_patches = [
        mpatches.Patch(color=BEST_COLOR,     label="Best RRF config ★"),
        mpatches.Patch(color=OTHER_COLOR,    label="Other RRF configs"),
        mpatches.Patch(color=_SOLO_BAR_COLOR, label="Solo model (best tuned)"),
    ]
    fig.legend(handles=legend_patches, loc="upper right", fontsize=8,
               framealpha=0.9, borderpad=0.8)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 0.86, 0.95])

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plots] Saved RRF sweep -> {save_path}")

    plt.close(fig)
    return fig


def plot_rrf_sweep(result, save_path: Optional[Path] = None) -> plt.Figure:
    """
    2×2 bar chart for RRF pair × k grid search. Accepts a SweepResult (kind="rrf").

    Solo-model scores from result.meta["solo_scores"] are injected as orange bars
    and sorted together with the RRF pair bars so the comparison is immediate.
    The best RRF config is still highlighted in seagreen; solo bars are orange.
    """
    rows = result.rows   # sorted by mean_ndcg desc from sweep
    solo_scores: dict = result.meta.get("solo_scores") or {}

    # Build unified entry list: RRF rows first, then solo entries
    entries = [
        {"label": r["label"], "ndcg": r["mean_ndcg"], "map": r["mean_map"],
         "mrr": r["mean_mrr"], "p10": r["mean_p10"], "is_solo": False}
        for r in rows
    ]
    for solo_name, sc in solo_scores.items():
        entries.append({
            "label":   f"{solo_name} best solo",
            "ndcg":    sc.get("ndcg", 0.0),
            "map":     sc.get("map",  0.0),
            "mrr":     sc.get("mrr",  0.0),
            "p10":     sc.get("p10",  0.0),
            "is_solo": True,
        })

    # Sort all together by NDCG descending
    entries.sort(key=lambda e: e["ndcg"], reverse=True)

    labels   = [e["label"]   for e in entries]
    ndcg     = [e["ndcg"]    for e in entries]
    maps     = [e["map"]     for e in entries]
    mrrs     = [e["mrr"]     for e in entries]
    p10s     = [e["p10"]     for e in entries]
    is_solo  = [e["is_solo"] for e in entries]

    # Best RRF index (first non-solo after sorting)
    best_rrf_idx = next(
        (i for i, s in enumerate(is_solo) if not s), 0
    )

    n_pairs = len(set(r["pair"] for r in rows))
    n_k     = len(set(r["rrf_k"] for r in rows))
    n_solo  = len(solo_scores)
    title   = (
        f"RRF Pair × k Grid Search — NDCG@100 / MAP / MRR / P@10 "
        f"({n_pairs} pairs × {n_k} k-values + {n_solo} solo refs, 5-fold CV on train)"
    )

    return plot_rrf_bars(
        labels       = labels,
        ndcg_values  = ndcg,
        map_values   = maps,
        mrr_values   = mrrs,
        p10_values   = p10s,
        best_rrf_idx = best_rrf_idx,
        is_solo      = is_solo,
        title        = title,
        save_path    = save_path,
    )


# ── Tuning summary: baseline vs best for all models ─────────────────────────

def plot_tuning_summary(
    bm25,
    lmjm,
    lmdir,
    enc_sweep,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Two-panel summary figure for the full Phase 1 tuning session.

    Left panel  — grouped bar chart: baseline NDCG@100 vs tuned NDCG@100 for every model.
    Right panel — horizontal gain bars: ΔNDCG@100 per model.

    Each SweepResult must expose .best (dict with 'mean_ndcg' or 'ndcg') and
    .baseline() (same structure, or None if no baseline in grid).
    """
    import pandas as pd

    # ── Collect numbers ───────────────────────────────────────────────────────
    def _ndcg(row, key="mean_ndcg"):
        """Accept both 'mean_ndcg' (CV sweeps) and 'ndcg' (encoder sweep)."""
        return row.get(key) or row.get("ndcg") or row.get("mean_ndcg") or 0.0

    entries = [
        ("BM25",    bm25.baseline(),    bm25.best),
        ("LM-JM",   lmjm.baseline(),    lmjm.best),
        ("LM-Dir",  lmdir.baseline(),   lmdir.best),
        ("Encoder", enc_sweep.baseline(), enc_sweep.best),
    ]

    labels    = [e[0] for e in entries]
    baselines = [_ndcg(e[1]) if e[1] else 0.0 for e in entries]
    bests     = [_ndcg(e[2]) for e in entries]
    deltas    = [b - a for a, b in zip(baselines, bests)]

    # ── Best-config strings for annotations ──────────────────────────────────
    def _cfg(entry_name, best_row):
        if entry_name == "BM25":
            return f"k1={best_row.get('k1', '?'):.2f}, b={best_row.get('b', '?'):.2f}"
        if entry_name == "LM-JM":
            return f"λ={best_row.get('lambda', '?'):.1f}"
        if entry_name == "LM-Dir":
            return f"μ={best_row.get('mu', '?')}"
        if entry_name == "Encoder":
            return str(best_row.get("alias", "?"))
        return "?"

    cfg_labels = [_cfg(e[0], e[2]) for e in entries]

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 4.5),
                                             gridspec_kw={"width_ratios": [2, 1]})

    x = np.arange(len(labels))
    w = 0.35

    bars_base = ax_left.bar(x - w / 2, baselines, w, color=BASELINE_COLOR,
                             label="Baseline", edgecolor="white")
    bars_best = ax_left.bar(x + w / 2, bests,     w, color=BEST_COLOR,
                             label="Tuned best", edgecolor="white")

    for bar, val in zip(bars_base, baselines):
        ax_left.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=7.5)
    for bar, val, cfg in zip(bars_best, bests, cfg_labels):
        ax_left.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                     f"{val:.4f}\n({cfg})", ha="center", va="bottom", fontsize=7,
                     color=BEST_COLOR, fontweight="bold")

    ax_left.set_xticks(x)
    ax_left.set_xticklabels(labels)
    ax_left.set_ylabel("NDCG@100 (train set)")
    ax_left.set_title("Baseline vs Tuned — NDCG@100", fontsize=11)
    ymax = max(bests) + 0.08
    ymin = max(0, min(baselines) - 0.05)
    ax_left.set_ylim(ymin, ymax)
    ax_left.legend(fontsize=9)

    # ── Right: delta bars ────────────────────────────────────────────────────
    delta_colors = [BEST_COLOR if d >= 0 else "tomato" for d in deltas]
    h_bars = ax_right.barh(labels, deltas, color=delta_colors, edgecolor="white")
    ax_right.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    for bar, d in zip(h_bars, deltas):
        sign = "+" if d >= 0 else ""
        ax_right.text(
            d + (0.0005 if d >= 0 else -0.0005),
            bar.get_y() + bar.get_height() / 2,
            f"{sign}{d:.4f}",
            ha="left" if d >= 0 else "right",
            va="center", fontsize=9, fontweight="bold",
            color=BEST_COLOR if d >= 0 else "tomato",
        )
    ax_right.set_xlabel("ΔNDCG@100")
    ax_right.set_title("Gain from tuning", fontsize=11)
    ax_right.invert_yaxis()

    fig.suptitle("Phase 1 — Hyperparameter Tuning Summary (train-set 5-fold CV)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"[plots] Saved tuning summary -> {save_path}")

    plt.close(fig)
    return fig


# ── PR curve interpretation helper ──────────────────────────────────────────

def plot_pr_interpretation(
    real_strategy_curves: dict,   # {name: (recall_array, precision_array)} -- our actual results
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    3x2 grid (6 panels) teaching how to read PR curves, then showing our real results.
      Row 1: perfect / random / good-retriever reference shapes
      Row 2: fair retriever / moderate retriever / our actual test-set results
    The bottom-left two panels sit in the same MAP range as our real models,
    so the reader can visually locate where our strategies fall on the quality scale.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    recall = np.linspace(0, 1, 50)

    # ── Row 1: three reference extremes ─────────────────────────────────────

    # panel 1 -- perfect: precision = 1.0 at all recall levels
    ax = axes[0, 0]
    ax.plot(recall, np.ones_like(recall), "b-o", markersize=3, label="Perfect (AUC=1.0)")
    ax.fill_between(recall, np.ones_like(recall), alpha=0.15)
    ax.set_title("Perfect retriever", fontsize=10)

    # panel 2 -- random: precision ~ fraction of relevant in corpus (1.1% for our dataset)
    ax = axes[0, 1]
    rnd = 0.011 * np.ones_like(recall)
    ax.plot(recall, rnd, "r--o", markersize=3, label="Random (MAP~0.01)")
    ax.fill_between(recall, rnd, alpha=0.12, color="red")
    ax.set_title("Random retriever (chance)", fontsize=10)

    # panel 3 -- typical strong IR: starts ~1.0 precision, drops slowly (concave)
    ax = axes[0, 2]
    good = 1.0 / (1 + 2.5 * recall)
    ax.plot(recall, good, "g-o", markersize=3, label="Good retriever (MAP~0.55)")
    ax.fill_between(recall, good, alpha=0.15, color="green")
    ax.set_title("Typical good retriever", fontsize=10)

    # ── Row 2: intermediate shapes + actual results ──────────────────────────

    # panel 4 -- fair retriever (MAP~0.25): drops steeply after first few relevant docs
    # this is around the range of a weak or domain-mismatched dense model
    ax = axes[1, 0]
    fair = np.where(recall < 0.25, 0.65 - recall * 1.8, np.maximum(0.011, 0.25 - (recall - 0.25) * 0.3))
    ax.plot(recall, fair, color="darkorange", marker="o", markersize=3, label="Fair retriever (MAP~0.25)")
    ax.fill_between(recall, fair, alpha=0.13, color="darkorange")
    ax.set_title("Fair retriever (near-diagonal)", fontsize=10)

    # panel 5 -- moderate retriever (MAP~0.45): between good and fair
    # similar shape to our best lexical-only models
    ax = axes[1, 1]
    moderate = 1.0 / (1 + 5.0 * recall ** 1.5)
    ax.plot(recall, moderate, color="mediumpurple", marker="o", markersize=3, label="Moderate retriever (MAP~0.45)")
    ax.fill_between(recall, moderate, alpha=0.13, color="mediumpurple")
    ax.set_title("Moderate retriever (similar range to our models)", fontsize=10)

    # panel 6 -- our actual results (all strategies overlaid)
    ax = axes[1, 2]
    _colors = {
        "BM25":            "steelblue",
        "LM-JM":           "tomato",
        "LM-Dir":          "seagreen",
        "KNN":             "darkorange",
        "RRF":             "purple",
        "BM25 (tuned)":    "steelblue",
        "LM-Dir (mu=75)":  "seagreen",
        "KNN (MedCPT)":    "gold",
        "RRF (tuned)":     "darkviolet",
    }
    for name, (rl, mp) in real_strategy_curves.items():
        color = _colors.get(name, "gray")
        ax.plot(rl, mp, "o-", color=color, markersize=2, linewidth=1.2, label=name)
        ax.fill_between(rl, mp, alpha=0.05, color=color)
    ax.set_title("Our actual results -- Phase 1 (33 test queries)", fontsize=10)
    ax.legend(loc="upper right", fontsize=6, ncol=2)
    ax.grid(alpha=0.3)

    # ── shared axis formatting ───────────────────────────────────────────────
    for row in axes:
        for ax in row:
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.05, 1.1)
            ax.set_xlabel("Recall", fontsize=8)
            ax.set_ylabel("Precision", fontsize=8)
            if ax != axes[1, 2]:   # legend already set for actual-results panel
                ax.legend(fontsize=8)
            ax.grid(alpha=0.25)

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
        ndcg_values=[0.6174, 0.7300, 0.6880],
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
