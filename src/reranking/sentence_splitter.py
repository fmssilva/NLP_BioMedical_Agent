"""
Sentence segmentation and selection for cross-encoder re-ranking.

* ``split_sentences`` – NLTK ``sent_tokenize`` wrapper with edge-case guards.
* ``select_top_sentences`` – split → score → return top-N dicts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import nltk

if TYPE_CHECKING:
    from src.reranking.cross_encoder import CrossEncoder


# ---------------------------------------------------------------------------
#  Sentence splitting
# ---------------------------------------------------------------------------
def split_sentences(text: str | None) -> List[str]:
    """Split *text* into sentences with NLTK punkt.

    Returns ``[]`` for ``None``, empty string, or whitespace-only input.
    Each returned sentence is stripped of leading/trailing whitespace;
    empty strings after stripping are discarded.
    """
    if not text or not text.strip():
        return []
    sentences = nltk.sent_tokenize(text.strip())
    return [s.strip() for s in sentences if s.strip()]


# ---------------------------------------------------------------------------
#  Top-N sentence selection
# ---------------------------------------------------------------------------
def select_top_sentences(
    query: str,
    abstract: str,
    cross_encoder: "CrossEncoder",
    top_n: int = 3,
) -> List[dict]:
    """Split abstract → score each sentence with cross-encoder → return top-N.

    Returns
    -------
    list[dict]
        ``[{"sentence": str, "score": float, "rank": int}, ...]``
        sorted by score **descending**.  ``rank`` is 1-based.
        If the abstract yields fewer than *top_n* sentences, all are returned.
        If the abstract is empty / None, returns ``[]``.
    """
    sentences = split_sentences(abstract)
    if not sentences:
        return []

    scores = cross_encoder.score_query_vs_sentences(query, sentences)

    # Pair up, sort descending by score
    paired = sorted(
        zip(sentences, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    top = paired[:top_n]
    return [
        {"sentence": sent, "score": sc, "rank": rank}
        for rank, (sent, sc) in enumerate(top, 1)
    ]
