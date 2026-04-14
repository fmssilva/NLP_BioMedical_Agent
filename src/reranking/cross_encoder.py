"""
Cross-encoder wrapper for sentence-level re-ranking.

Loads a HuggingFace ``AutoModelForSequenceClassification`` checkpoint
(e.g. ``ncbi/MedCPT-Cross-Encoder``) and exposes two scoring methods:

* ``score_pairs``  – low-level: score arbitrary (text_a, text_b) pairs.
* ``score_query_vs_sentences`` – high-level: score one query against many sentences.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton cache  (same idea as src.embeddings.encoder.Encoder)
# ---------------------------------------------------------------------------
_CACHE: dict[str, "CrossEncoder"] = {}


class CrossEncoder:
    """Load and cache a HuggingFace cross-encoder model."""

    # ------------------------------------------------------------------ #
    #  Construction / Singleton
    # ------------------------------------------------------------------ #
    def __new__(cls, model_name: str, device: str | None = None):
        key = f"{model_name}@{device or 'auto'}"
        if key not in _CACHE:
            instance = super().__new__(cls)
            instance._initialised = False
            _CACHE[key] = instance
        return _CACHE[key]

    def __init__(self, model_name: str, device: str | None = None):
        if self._initialised:
            return
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device).eval()

        # Detect number of output labels (1 = single logit, 2 = [irr, rel])
        self.num_labels: int = self.model.config.num_labels
        logger.info(
            "CrossEncoder loaded: %s  |  num_labels=%d  |  device=%s",
            model_name, self.num_labels, self.device,
        )
        self._initialised = True

    # ------------------------------------------------------------------ #
    #  Low-level: score arbitrary (text_a, text_b) pairs
    # ------------------------------------------------------------------ #
    def score_pairs(
        self,
        pairs: List[Tuple[str, str]],
        batch_size: int = 16,
    ) -> List[float]:
        """Return a list of float logit scores, one per input pair.

        If the model outputs 2-class logits (``[irrelevant, relevant]``),
        the *relevant* class logit (index 1) is returned.  If the model
        outputs a single logit, that value is returned directly.
        """
        scores: list[float] = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            texts_a = [p[0] for p in batch]
            texts_b = [p[1] for p in batch]
            encoded = self.tokenizer(
                texts_a,
                texts_b,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**encoded).logits  # (B, num_labels)
            if self.num_labels == 1:
                batch_scores = logits.squeeze(-1).cpu().numpy()
            else:
                # 2-class: take relevant-class logit (index 1)
                batch_scores = logits[:, 1].cpu().numpy()
            scores.extend(batch_scores.tolist())
        return scores

    # ------------------------------------------------------------------ #
    #  High-level: one query vs many sentences
    # ------------------------------------------------------------------ #
    def score_query_vs_sentences(
        self,
        query: str,
        sentences: List[str],
        batch_size: int = 16,
    ) -> List[float]:
        """Score *query* against each sentence.  Returns aligned list of floats."""
        if not sentences:
            return []
        pairs = [(query, s) for s in sentences]
        return self.score_pairs(pairs, batch_size=batch_size)

    # ------------------------------------------------------------------ #
    #  Repr
    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:
        return f"CrossEncoder({self.model_name!r}, device={self.device!r})"
