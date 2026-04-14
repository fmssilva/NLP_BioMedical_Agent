"""
Tests for src/analysis — transformer_inspector.py and attention_plots.py.

Run with:  pytest src/analysis/__analysis_test.py -v
Uses bert-base-uncased (small enough for CPU).
The model is loaded once per session via a module-level fixture.
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for testing
import matplotlib.pyplot as plt

from transformers import AutoModel, AutoTokenizer

# ── Module-level fixtures (load model ONCE) ──────────────────────────────────

_MODEL_NAME = "bert-base-uncased"
_tokenizer = None
_model = None


def _ensure_model():
    """Lazy-load the model once for the entire test module."""
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        _model = AutoModel.from_pretrained(
            _MODEL_NAME,
            output_hidden_states=True,
            output_attentions=True,
        )
        _model.eval()
    return _tokenizer, _model


# ═════════════════════════════════════════════════════════════════════════════
# Group 1: transformer_inspector — get_hidden_states_and_attentions
# ═════════════════════════════════════════════════════════════════════════════

class TestGetHiddenStatesAndAttentions:
    """Tests for get_hidden_states_and_attentions()."""

    def test_returns_correct_types(self):
        from src.analysis.transformer_inspector import get_hidden_states_and_attentions
        tok, mdl = _ensure_model()
        tokens, hs, att = get_hidden_states_and_attentions("Hello world.", tok, mdl)
        assert isinstance(tokens, list)
        assert all(isinstance(t, str) for t in tokens)
        assert isinstance(hs, np.ndarray)
        assert isinstance(att, np.ndarray)

    def test_hidden_states_shape(self):
        """hidden_states should be (n_layers+1, seq_len, hidden_dim)."""
        from src.analysis.transformer_inspector import get_hidden_states_and_attentions
        tok, mdl = _ensure_model()
        tokens, hs, att = get_hidden_states_and_attentions("Hello world.", tok, mdl)
        n_layers = mdl.config.num_hidden_layers  # 12 for bert-base
        hidden_dim = mdl.config.hidden_size  # 768
        seq_len = len(tokens)
        assert hs.shape == (n_layers + 1, seq_len, hidden_dim)

    def test_attentions_shape(self):
        """attentions should be (n_layers, n_heads, seq_len, seq_len)."""
        from src.analysis.transformer_inspector import get_hidden_states_and_attentions
        tok, mdl = _ensure_model()
        tokens, hs, att = get_hidden_states_and_attentions("Hello world.", tok, mdl)
        n_layers = mdl.config.num_hidden_layers
        n_heads = mdl.config.num_attention_heads  # 12 for bert-base
        seq_len = len(tokens)
        assert att.shape == (n_layers, n_heads, seq_len, seq_len)

    def test_tokens_include_special(self):
        """Token list should contain [CLS] and [SEP]."""
        from src.analysis.transformer_inspector import get_hidden_states_and_attentions
        tok, mdl = _ensure_model()
        tokens, _, _ = get_hidden_states_and_attentions("Hello world.", tok, mdl)
        assert tokens[0] == "[CLS]"
        assert "[SEP]" in tokens

    def test_attention_rows_sum_to_one(self):
        """Each attention row (per head, per layer) should sum to ~1.0."""
        from src.analysis.transformer_inspector import get_hidden_states_and_attentions
        tok, mdl = _ensure_model()
        _, _, att = get_hidden_states_and_attentions("Test sentence.", tok, mdl)
        row_sums = att.sum(axis=-1)  # (L, H, S)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-4)

    def test_longer_input(self):
        """Longer input should produce larger seq_len."""
        from src.analysis.transformer_inspector import get_hidden_states_and_attentions
        tok, mdl = _ensure_model()
        short_tokens, _, _ = get_hidden_states_and_attentions("Hi.", tok, mdl)
        long_tokens, _, _ = get_hidden_states_and_attentions(
            "The quick brown fox jumps over the lazy dog multiple times.", tok, mdl
        )
        assert len(long_tokens) > len(short_tokens)


# ═════════════════════════════════════════════════════════════════════════════
# Group 2: transformer_inspector — get_positional_embeddings
# ═════════════════════════════════════════════════════════════════════════════

class TestGetPositionalEmbeddings:
    """Tests for get_positional_embeddings()."""

    def test_returns_correct_types(self):
        from src.analysis.transformer_inspector import get_positional_embeddings
        tok, mdl = _ensure_model()
        tokens, emb = get_positional_embeddings(tok, mdl, word="the", n=50)
        assert isinstance(tokens, list)
        assert isinstance(emb, np.ndarray)

    def test_embedding_shape(self):
        """Embeddings should be (n_tokens, hidden_dim)."""
        from src.analysis.transformer_inspector import get_positional_embeddings
        tok, mdl = _ensure_model()
        tokens, emb = get_positional_embeddings(tok, mdl, word="the", n=50)
        assert emb.ndim == 2
        assert emb.shape[0] == len(tokens)
        assert emb.shape[1] == mdl.config.hidden_size

    def test_tokens_are_same_word(self):
        """All non-special tokens should be the same word."""
        from src.analysis.transformer_inspector import get_positional_embeddings
        tok, mdl = _ensure_model()
        tokens, _ = get_positional_embeddings(tok, mdl, word="the", n=50)
        content_tokens = [t for t in tokens if t not in ("[CLS]", "[SEP]")]
        assert len(set(content_tokens)) == 1
        assert content_tokens[0] == "the"

    def test_truncation_at_max_length(self):
        """Requesting n=10000 should be truncated to model max_length."""
        from src.analysis.transformer_inspector import get_positional_embeddings
        tok, mdl = _ensure_model()
        tokens, emb = get_positional_embeddings(tok, mdl, word="the", n=10000)
        max_len = tok.model_max_length
        assert len(tokens) <= max_len
        assert emb.shape[0] <= max_len

    def test_different_word(self):
        """Should work with different neutral words."""
        from src.analysis.transformer_inspector import get_positional_embeddings
        tok, mdl = _ensure_model()
        tokens, emb = get_positional_embeddings(tok, mdl, word="cat", n=30)
        content_tokens = [t for t in tokens if t not in ("[CLS]", "[SEP]")]
        assert content_tokens[0] == "cat"
        assert emb.shape[0] == len(tokens)


# ═════════════════════════════════════════════════════════════════════════════
# Group 3: transformer_inspector — cosine_distance_matrix
# ═════════════════════════════════════════════════════════════════════════════

class TestCosineDistanceMatrix:
    """Tests for cosine_distance_matrix()."""

    def test_shape_is_square(self):
        from src.analysis.transformer_inspector import cosine_distance_matrix
        emb = np.random.randn(20, 64).astype(np.float32)
        dm = cosine_distance_matrix(emb)
        assert dm.shape == (20, 20)

    def test_diagonal_is_zero(self):
        from src.analysis.transformer_inspector import cosine_distance_matrix
        emb = np.random.randn(10, 64).astype(np.float32)
        dm = cosine_distance_matrix(emb)
        np.testing.assert_allclose(np.diag(dm), 0.0, atol=1e-6)

    def test_symmetric(self):
        from src.analysis.transformer_inspector import cosine_distance_matrix
        emb = np.random.randn(15, 64).astype(np.float32)
        dm = cosine_distance_matrix(emb)
        np.testing.assert_allclose(dm, dm.T, atol=1e-6)

    def test_non_negative(self):
        from src.analysis.transformer_inspector import cosine_distance_matrix
        emb = np.random.randn(10, 64).astype(np.float32)
        dm = cosine_distance_matrix(emb)
        assert (dm >= -1e-6).all(), "Distance matrix should be non-negative"

    def test_identical_vectors_have_zero_distance(self):
        from src.analysis.transformer_inspector import cosine_distance_matrix
        emb = np.tile(np.array([1.0, 2.0, 3.0]), (5, 1))
        dm = cosine_distance_matrix(emb)
        np.testing.assert_allclose(dm, 0.0, atol=1e-6)

    def test_orthogonal_vectors_have_distance_one(self):
        from src.analysis.transformer_inspector import cosine_distance_matrix
        emb = np.eye(3, dtype=np.float32)  # 3 orthogonal unit vectors
        dm = cosine_distance_matrix(emb)
        # Off-diagonal should be 1.0
        for i in range(3):
            for j in range(3):
                if i != j:
                    np.testing.assert_allclose(dm[i, j], 1.0, atol=1e-6)


# ═════════════════════════════════════════════════════════════════════════════
# Group 4: attention_plots — plot_positional_embedding_scatter
# ═════════════════════════════════════════════════════════════════════════════

class TestPlotPositionalEmbeddingScatter:
    """Tests for plot_positional_embedding_scatter()."""

    def test_returns_figure(self):
        from src.analysis.attention_plots import plot_positional_embedding_scatter
        emb = np.random.randn(50, 64).astype(np.float32)
        fig = plot_positional_embedding_scatter(emb)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_has_axes_with_labels(self):
        from src.analysis.attention_plots import plot_positional_embedding_scatter
        emb = np.random.randn(50, 64).astype(np.float32)
        fig = plot_positional_embedding_scatter(emb)
        ax = fig.axes[0]
        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""
        assert ax.get_title() != ""
        plt.close(fig)

    def test_save_path(self, tmp_path):
        from src.analysis.attention_plots import plot_positional_embedding_scatter
        emb = np.random.randn(30, 64).astype(np.float32)
        out = str(tmp_path / "test_scatter.png")
        fig = plot_positional_embedding_scatter(emb, save_path=out)
        import os
        assert os.path.exists(out)
        plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# Group 5: attention_plots — plot_pairwise_distance_heatmap
# ═════════════════════════════════════════════════════════════════════════════

class TestPlotPairwiseDistanceHeatmap:
    """Tests for plot_pairwise_distance_heatmap()."""

    def test_returns_figure(self):
        from src.analysis.attention_plots import plot_pairwise_distance_heatmap
        dm = np.random.rand(20, 20).astype(np.float32)
        fig = plot_pairwise_distance_heatmap(dm)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_has_title(self):
        from src.analysis.attention_plots import plot_pairwise_distance_heatmap
        dm = np.random.rand(20, 20).astype(np.float32)
        fig = plot_pairwise_distance_heatmap(dm)
        ax = fig.axes[0]
        assert ax.get_title() != ""
        plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# Group 6: attention_plots — plot_contextual_embeddings_grid
# ═════════════════════════════════════════════════════════════════════════════

class TestPlotContextualEmbeddingsGrid:
    """Tests for plot_contextual_embeddings_grid()."""

    def test_returns_figure(self):
        from src.analysis.attention_plots import plot_contextual_embeddings_grid
        # Fake: 13 layers (embed + 12), 5 tokens, 64 dim
        hs = np.random.randn(13, 5, 64).astype(np.float32)
        tokens = ["[CLS]", "hello", "world", "!", "[SEP]"]
        fig = plot_contextual_embeddings_grid(hs, tokens)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_has_12_visible_subplots(self):
        from src.analysis.attention_plots import plot_contextual_embeddings_grid
        hs = np.random.randn(13, 5, 64).astype(np.float32)
        tokens = ["[CLS]", "hello", "world", "!", "[SEP]"]
        fig = plot_contextual_embeddings_grid(hs, tokens)
        visible = [ax for ax in fig.axes if ax.get_visible()]
        assert len(visible) == 12
        plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# Group 7: attention_plots — plot_attention_matrix
# ═════════════════════════════════════════════════════════════════════════════

class TestPlotAttentionMatrix:
    """Tests for plot_attention_matrix()."""

    def test_returns_figure(self):
        from src.analysis.attention_plots import plot_attention_matrix
        # 12 layers, 12 heads, 5 tokens
        att = np.random.rand(12, 12, 5, 5).astype(np.float32)
        tokens = ["[CLS]", "hello", "world", "!", "[SEP]"]
        fig = plot_attention_matrix(att, tokens, layer=11)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_different_layers(self):
        from src.analysis.attention_plots import plot_attention_matrix
        att = np.random.rand(12, 12, 5, 5).astype(np.float32)
        tokens = ["[CLS]", "hello", "world", "!", "[SEP]"]
        for layer in [0, 5, 11]:
            fig = plot_attention_matrix(att, tokens, layer=layer)
            assert isinstance(fig, plt.Figure)
            plt.close(fig)

    def test_title_suffix(self):
        from src.analysis.attention_plots import plot_attention_matrix
        att = np.random.rand(12, 12, 5, 5).astype(np.float32)
        tokens = ["[CLS]", "a", "b", "c", "[SEP]"]
        fig = plot_attention_matrix(att, tokens, layer=11, title_suffix="(relevant)")
        ax = fig.axes[0]
        assert "(relevant)" in ax.get_title()
        plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# Group 8: Integration — full pipeline with real model
# ═════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    """End-to-end integration: model → inspector → plots."""

    def test_positional_pipeline(self):
        """get_positional_embeddings → cosine_distance_matrix → scatter + heatmap."""
        from src.analysis.transformer_inspector import get_positional_embeddings, cosine_distance_matrix
        from src.analysis.attention_plots import plot_positional_embedding_scatter, plot_pairwise_distance_heatmap
        tok, mdl = _ensure_model()

        tokens, emb = get_positional_embeddings(tok, mdl, word="the", n=50)
        assert emb.shape[0] == len(tokens)

        dm = cosine_distance_matrix(emb)
        assert dm.shape[0] == dm.shape[1] == len(tokens)

        fig1 = plot_positional_embedding_scatter(emb)
        assert isinstance(fig1, plt.Figure)
        plt.close(fig1)

        fig2 = plot_pairwise_distance_heatmap(dm)
        assert isinstance(fig2, plt.Figure)
        plt.close(fig2)

    def test_contextual_pipeline(self):
        """get_hidden_states_and_attentions → contextual grid + attention heatmap."""
        from src.analysis.transformer_inspector import get_hidden_states_and_attentions
        from src.analysis.attention_plots import plot_contextual_embeddings_grid, plot_attention_matrix
        tok, mdl = _ensure_model()

        tokens, hs, att = get_hidden_states_and_attentions(
            "The drug treatment showed significant effects.", tok, mdl
        )
        assert hs.shape[0] == 13  # 12 layers + embedding layer

        fig1 = plot_contextual_embeddings_grid(hs, tokens)
        assert isinstance(fig1, plt.Figure)
        plt.close(fig1)

        fig2 = plot_attention_matrix(att, tokens, layer=11)
        assert isinstance(fig2, plt.Figure)
        plt.close(fig2)

    def test_positional_distance_gradient(self):
        """Nearby positions should have lower distance than far positions (on average)."""
        from src.analysis.transformer_inspector import get_positional_embeddings, cosine_distance_matrix
        tok, mdl = _ensure_model()

        _, emb = get_positional_embeddings(tok, mdl, word="the", n=100)
        dm = cosine_distance_matrix(emb)

        # Average distance for offset=1 vs offset=50
        n = dm.shape[0]
        near_dists = [dm[i, i + 1] for i in range(n - 1)]
        far_dists = [dm[i, min(i + 50, n - 1)] for i in range(n - 50)]

        avg_near = np.mean(near_dists)
        avg_far = np.mean(far_dists)
        assert avg_near < avg_far, (
            f"Near positions should be closer: avg_near={avg_near:.4f}, avg_far={avg_far:.4f}"
        )
