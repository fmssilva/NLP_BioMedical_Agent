"""
Transformer inspector utilities for Phase 2 — Section 2: Transformer Internals.

Provides functions to extract hidden states, attentions, and positional embeddings
from BERT-family models, and to compute cosine distance matrices for analysis.
"""

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_distances


def get_hidden_states_and_attentions(
    text: str,
    tokenizer,
    model,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    Run a forward pass and extract hidden states + attention weights.

    Parameters
    ----------
    text : str
        Input text (or ``[SEP]``-separated query+doc for cross-encoders).
    tokenizer : PreTrainedTokenizer
    model : PreTrainedModel
        Must have been loaded with ``output_hidden_states=True``
        and ``output_attentions=True``.

    Returns
    -------
    tokens : list[str]
        Decoded token strings (length = seq_len, includes [CLS]/[SEP]).
    hidden_states : np.ndarray, shape (n_layers+1, seq_len, hidden_dim)
        Layer 0 = embedding layer; layers 1..n_layers = transformer blocks.
    attentions : np.ndarray, shape (n_layers, n_heads, seq_len, seq_len)
        Attention weights for each layer and head.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Decode tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # hidden_states: tuple of (n_layers+1) tensors, each (1, seq_len, hidden_dim)
    hidden_states = torch.stack(outputs.hidden_states, dim=0)  # (L+1, 1, S, H)
    hidden_states = hidden_states.squeeze(1).cpu().numpy()      # (L+1, S, H)

    # attentions: tuple of n_layers tensors, each (1, n_heads, seq_len, seq_len)
    attentions = torch.stack(outputs.attentions, dim=0)  # (L, 1, H, S, S)
    attentions = attentions.squeeze(1).cpu().numpy()      # (L, H, S, S)

    return tokens, hidden_states, attentions


def get_positional_embeddings(
    tokenizer,
    model,
    word: str = "the",
    n: int = 200,
) -> tuple[list[str], np.ndarray]:
    """
    Isolate positional embeddings by feeding the *same word* repeated ``n`` times.

    Because every token is the same word, the only variation in the layer-0
    (embedding layer) output comes from the learned positional embeddings.
    The input is truncated to the model's ``max_position_embeddings``.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizer
    model : PreTrainedModel
        Must have ``output_hidden_states=True``.
    word : str
        Neutral word to repeat (default ``"the"``).
    n : int
        Number of repetitions (will be capped at model max_length - 2 for
        [CLS] and [SEP]).

    Returns
    -------
    tokens : list[str]
        Decoded token strings (includes [CLS] and [SEP]).
    embeddings : np.ndarray, shape (n_tokens, hidden_dim)
        Layer-0 hidden states (embedding layer output).
    """
    # Build repeated text; tokenizer will truncate to max_length
    text = " ".join([word] * n)
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # Layer 0 = embedding layer output (before any transformer block)
    embeddings = outputs.hidden_states[0].squeeze(0).cpu().numpy()  # (S, H)

    return tokens, embeddings


def cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine distances between all embedding vectors.

    Parameters
    ----------
    embeddings : np.ndarray, shape (N, D)

    Returns
    -------
    dist_matrix : np.ndarray, shape (N, N)
        ``dist_matrix[i, j] = 1 - cosine_similarity(embeddings[i], embeddings[j])``
    """
    return cosine_distances(embeddings)


# ── Quick local testing ──────────────────────────────────────────────────────
if __name__ == "__main__":
    from transformers import AutoModel, AutoTokenizer

    model_name = "bert-base-uncased"
    print(f"Loading {model_name} ...")
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(
        model_name,
        output_hidden_states=True,
        output_attentions=True,
    )
    mdl.eval()

    # Test get_positional_embeddings
    tokens, emb = get_positional_embeddings(tok, mdl, word="the", n=200)
    print(f"Positional embeddings: tokens={len(tokens)}, shape={emb.shape}")

    # Test get_hidden_states_and_attentions
    tokens2, hs, att = get_hidden_states_and_attentions(
        "The drug treatment showed significant effects.", tok, mdl,
    )
    print(f"Hidden states: tokens={len(tokens2)}, hs={hs.shape}, att={att.shape}")

    # Test cosine_distance_matrix
    dm = cosine_distance_matrix(emb)
    print(f"Distance matrix: shape={dm.shape}")
    print("All tests passed.")
