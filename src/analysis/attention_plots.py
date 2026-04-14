"""
Attention and embedding visualisation functions for Phase 2 — Section 2.

All functions return a ``matplotlib.figure.Figure`` so the caller can
``plt.show()`` or ``fig.savefig()`` as needed.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from src.analysis.transformer_inspector import cosine_distance_matrix


# ─────────────────────────────────────────────────────────────────────────────
# Plot C — Positional Embedding Scatter (2D PCA, colour = distance from token 0)
# ─────────────────────────────────────────────────────────────────────────────
def plot_positional_embedding_scatter(
    embeddings: np.ndarray,
    save_path: str | None = None,
) -> plt.Figure:
    """
    PCA-project positional embeddings to 2D and colour-code by cosine
    distance from token 0.

    Parameters
    ----------
    embeddings : np.ndarray, shape (N, D)
        Layer-0 hidden states (one row per position).
    save_path : str | None
        If given, saves the figure to this path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # Distances from token 0
    dist_from_0 = cosine_distance_matrix(embeddings)[0]  # row 0 -> (N,)

    # PCA to 2D
    coords_2d = PCA(n_components=2).fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(
        coords_2d[:, 0],
        coords_2d[:, 1],
        c=dist_from_0,
        cmap="viridis",
        s=20,
        alpha=0.8,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Cosine distance from token 0", fontsize=11)
    ax.set_xlabel("PC 1", fontsize=12)
    ax.set_ylabel("PC 2", fontsize=12)
    ax.set_title("Positional Embeddings — PCA 2D (colour = distance from position 0)", fontsize=13)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Plot D — Pairwise Distance Heatmap
# ─────────────────────────────────────────────────────────────────────────────
def plot_pairwise_distance_heatmap(
    distance_matrix: np.ndarray,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Seaborn heatmap of pairwise cosine distance matrix.

    Parameters
    ----------
    distance_matrix : np.ndarray, shape (N, N)
    save_path : str | None

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        distance_matrix,
        cmap="viridis",
        ax=ax,
        xticklabels=50,
        yticklabels=50,
        cbar_kws={"label": "Cosine distance"},
    )
    ax.set_xlabel("Token position", fontsize=12)
    ax.set_ylabel("Token position", fontsize=12)
    ax.set_title("Positional Embeddings — Pairwise Cosine Distance Matrix", fontsize=13)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Plot E — Contextual Embeddings Grid (12 layers, PCA 2D)
# ─────────────────────────────────────────────────────────────────────────────
def plot_contextual_embeddings_grid(
    hidden_states: np.ndarray,
    tokens: list[str],
    save_path: str | None = None,
) -> plt.Figure:
    """
    3×4 grid of PCA scatter plots — one per transformer layer (1..12).

    PCA is fitted on the **last layer** so all subplots share the same
    projection space, making cross-layer comparison meaningful.

    Parameters
    ----------
    hidden_states : np.ndarray, shape (n_layers+1, seq_len, hidden_dim)
        Layer 0 = embedding layer.  Layers 1..12 are plotted.
    tokens : list[str]
        Token labels for annotation.
    save_path : str | None

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    n_layers = hidden_states.shape[0] - 1  # exclude embedding layer
    rows, cols = 3, 4
    if n_layers != 12:
        # Adjust grid for non-12-layer models
        cols = 4
        rows = (n_layers + cols - 1) // cols

    # Fit PCA on the last layer for a common space
    pca = PCA(n_components=2)
    pca.fit(hidden_states[-1])  # last layer

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    axes = axes.flatten()

    for layer_idx in range(1, n_layers + 1):
        ax = axes[layer_idx - 1]
        coords = pca.transform(hidden_states[layer_idx])
        ax.scatter(coords[:, 0], coords[:, 1], s=40, alpha=0.7)
        # Annotate tokens (skip [CLS] and [SEP] labels for clarity if many)
        for i, tok in enumerate(tokens):
            ax.annotate(
                tok,
                (coords[i, 0], coords[i, 1]),
                fontsize=7,
                alpha=0.8,
                ha="center",
                va="bottom",
            )
        ax.set_title(f"Layer {layer_idx}", fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=7)

    # Hide unused axes
    for idx in range(n_layers, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        "Contextual Embeddings — Layer-by-Layer Evolution (PCA 2D, fitted on last layer)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Plot F/G — Attention Matrix Heatmap (mean across heads, one layer)
# ─────────────────────────────────────────────────────────────────────────────
def plot_attention_matrix(
    attentions: np.ndarray,
    tokens: list[str],
    layer: int = 11,
    save_path: str | None = None,
    title_suffix: str = "",
) -> plt.Figure:
    """
    Heatmap of attention weights averaged across heads for a single layer.

    Parameters
    ----------
    attentions : np.ndarray, shape (n_layers, n_heads, seq_len, seq_len)
    tokens : list[str]
        Token labels for axes.
    layer : int
        Which layer to visualise (0-indexed, default 11 = last layer).
    save_path : str | None
    title_suffix : str
        Extra text appended to the title (e.g. "(relevant)" or "(irrelevant)").

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # Mean across heads -> (seq_len, seq_len)
    mean_attn = attentions[layer].mean(axis=0)

    fig, ax = plt.subplots(figsize=(max(8, len(tokens) * 0.5), max(6, len(tokens) * 0.4)))
    sns.heatmap(
        mean_attn,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="Blues",
        ax=ax,
        vmin=0.0,
        cbar_kws={"label": "Attention weight"},
    )
    ax.set_xlabel("Key (attended to)", fontsize=11)
    ax.set_ylabel("Query (attending from)", fontsize=11)
    title = f"Self-Attention — Layer {layer + 1} (mean across heads)"
    if title_suffix:
        title += f"  {title_suffix}"
    ax.set_title(title, fontsize=12)
    plt.xticks(rotation=60, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
