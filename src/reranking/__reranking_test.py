"""
Comprehensive tests for ``src.reranking`` — cross-encoder + sentence splitter.

Tests cover:
  Group 1  CrossEncoder – load, score_pairs, score_query_vs_sentences, batching, logit shape
  Group 2  split_sentences – edge cases, abbreviations, normal input
  Group 3  select_top_sentences – ranking, top-N, empty input
  Group 4  Integration – full pipeline (load model → split → score → select)
  Group 5  Notebook-cell logic – replicates key assertions the notebook will run

Run:
    python -m pytest src/reranking/__reranking_test.py -v --tb=short
"""

from __future__ import annotations

import pytest
import numpy as np

# ── Ensure project root on sys.path ────────────────────────────────────────
import sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Imports under test ──────────────────────────────────────────────────────
from src.reranking.sentence_splitter import split_sentences, select_top_sentences
from src.reranking.cross_encoder import CrossEncoder

# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------

# Use a small BERT-based cross-encoder for fast CPU tests.
# MedCPT-Cross-Encoder is ~440 MB – too slow for rapid iteration.
# Instead we use cross-encoder/ms-marco-TinyBERT-L-2-v2 (small, 1-label).
# If you want to test with MedCPT specifically, set the env var:
#   CE_TEST_MODEL=ncbi/MedCPT-Cross-Encoder
import os

_CE_MODEL = os.environ.get("CE_TEST_MODEL", "cross-encoder/ms-marco-TinyBERT-L-2-v2")


@pytest.fixture(scope="module")
def ce() -> CrossEncoder:
    """Module-scoped cross-encoder (loaded once for all tests)."""
    return CrossEncoder(_CE_MODEL, device="cpu")


# A realistic biomedical abstract with multiple sentences.
ABSTRACT = (
    "Obstructive sleep apnea (OSA) is a common disorder affecting approximately 10% of adults. "
    "Continuous positive airway pressure (CPAP) remains the gold standard treatment. "
    "Recent studies suggest that CPAP adherence significantly reduces cardiovascular risk. "
    "However, patient compliance remains a major clinical challenge. "
    "Alternative therapies, including oral appliances and positional therapy, show promise."
)

QUERY = "What is the treatment for obstructive sleep apnea?"


# ===========================================================================
#  Group 1: CrossEncoder
# ===========================================================================
class TestCrossEncoderLoad:
    """Model loads and has expected attributes."""

    def test_loads_without_error(self, ce: CrossEncoder):
        assert ce.model is not None
        assert ce.tokenizer is not None

    def test_has_num_labels(self, ce: CrossEncoder):
        assert isinstance(ce.num_labels, int)
        assert ce.num_labels in (1, 2), f"Unexpected num_labels={ce.num_labels}"

    def test_singleton_returns_same_instance(self):
        a = CrossEncoder(_CE_MODEL, device="cpu")
        b = CrossEncoder(_CE_MODEL, device="cpu")
        assert a is b

    def test_repr(self, ce: CrossEncoder):
        r = repr(ce)
        assert "CrossEncoder" in r
        assert _CE_MODEL in r


class TestScorePairs:
    """score_pairs returns correct types and shapes."""

    def test_single_pair(self, ce: CrossEncoder):
        result = ce.score_pairs([("sleep apnea", "CPAP is effective.")])
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], float)

    def test_multiple_pairs(self, ce: CrossEncoder):
        pairs = [
            ("sleep apnea", "CPAP is effective."),
            ("heart disease", "Exercise reduces risk."),
            ("cancer treatment", "Immunotherapy shows promise."),
            ("diabetes", "Insulin is required."),
            ("headache", "Take aspirin."),
        ]
        result = ce.score_pairs(pairs, batch_size=2)
        assert len(result) == 5
        assert all(isinstance(s, float) for s in result)

    def test_order_preserved(self, ce: CrossEncoder):
        """Batching must NOT re-order results."""
        pairs = [
            ("sleep apnea", "CPAP is effective for sleep apnea treatment."),
            ("cancer", "This fruit is called an apple."),
        ]
        scores = ce.score_pairs(pairs, batch_size=1)
        assert len(scores) == 2
        # Relevant pair should score higher than random pair
        assert scores[0] > scores[1], "Relevant pair should score higher"

    def test_empty_input(self, ce: CrossEncoder):
        result = ce.score_pairs([])
        assert result == []


class TestScoreQueryVsSentences:
    """score_query_vs_sentences convenience method."""

    def test_returns_aligned_scores(self, ce: CrossEncoder):
        sents = ["CPAP is effective.", "The weather is nice.", "Sleep apnea is common."]
        scores = ce.score_query_vs_sentences("sleep apnea treatment", sents)
        assert len(scores) == len(sents)

    def test_empty_sentences(self, ce: CrossEncoder):
        assert ce.score_query_vs_sentences("query", []) == []

    def test_single_sentence(self, ce: CrossEncoder):
        scores = ce.score_query_vs_sentences("test", ["Hello world."])
        assert len(scores) == 1
        assert isinstance(scores[0], float)


# ===========================================================================
#  Group 2: split_sentences
# ===========================================================================
class TestSplitSentences:
    """Edge cases and normal operation for NLTK-based splitting."""

    def test_empty_string(self):
        assert split_sentences("") == []

    def test_none_input(self):
        assert split_sentences(None) == []

    def test_whitespace_only(self):
        assert split_sentences("   \n\t  ") == []

    def test_single_sentence(self):
        result = split_sentences("One sentence.")
        assert result == ["One sentence."]

    def test_multiple_sentences(self):
        result = split_sentences("First. Second. Third.")
        assert len(result) == 3

    def test_biomedical_abbreviations(self):
        """Abbreviations like 'e.g.' and 'i.v.' should not over-split."""
        text = "The drug was administered i.v. in the trial. Results were significant."
        result = split_sentences(text)
        # Should be 2 sentences, not 3+
        assert len(result) <= 3, f"Over-split: got {len(result)} sentences"
        assert len(result) >= 2

    def test_no_period_title_only(self):
        """Title-only text with no period."""
        result = split_sentences("Impact of CPAP on cardiovascular outcomes")
        assert len(result) >= 1  # At least the text itself is returned

    def test_strips_whitespace(self):
        result = split_sentences("  Hello world.   Foo bar.  ")
        assert all(s == s.strip() for s in result)

    def test_real_abstract(self):
        result = split_sentences(ABSTRACT)
        assert len(result) == 5, f"Expected 5 sentences, got {len(result)}"

    def test_newlines_in_text(self):
        text = "First sentence.\nSecond sentence.\nThird sentence."
        result = split_sentences(text)
        assert len(result) == 3


# ===========================================================================
#  Group 3: select_top_sentences
# ===========================================================================
class TestSelectTopSentences:
    """Top-N selection with cross-encoder."""

    def test_returns_top_n(self, ce: CrossEncoder):
        result = select_top_sentences(QUERY, ABSTRACT, ce, top_n=3)
        assert len(result) == 3

    def test_sorted_descending(self, ce: CrossEncoder):
        result = select_top_sentences(QUERY, ABSTRACT, ce, top_n=5)
        scores = [r["score"] for r in result]
        assert scores == sorted(scores, reverse=True), "Should be sorted descending"

    def test_top1_ge_top3(self, ce: CrossEncoder):
        result = select_top_sentences(QUERY, ABSTRACT, ce, top_n=3)
        assert result[0]["score"] >= result[-1]["score"]

    def test_returns_fewer_if_short_abstract(self, ce: CrossEncoder):
        """Abstract with only 1 sentence → return 1, not crash."""
        result = select_top_sentences(QUERY, "CPAP is effective.", ce, top_n=3)
        assert len(result) == 1
        assert result[0]["rank"] == 1

    def test_empty_abstract(self, ce: CrossEncoder):
        result = select_top_sentences(QUERY, "", ce, top_n=3)
        assert result == []

    def test_none_abstract(self, ce: CrossEncoder):
        result = select_top_sentences(QUERY, None, ce, top_n=3)
        assert result == []

    def test_result_dict_structure(self, ce: CrossEncoder):
        result = select_top_sentences(QUERY, ABSTRACT, ce, top_n=2)
        for item in result:
            assert "sentence" in item
            assert "score" in item
            assert "rank" in item
            assert isinstance(item["sentence"], str)
            assert isinstance(item["score"], float)
            assert isinstance(item["rank"], int)

    def test_rank_is_one_based(self, ce: CrossEncoder):
        result = select_top_sentences(QUERY, ABSTRACT, ce, top_n=3)
        ranks = [r["rank"] for r in result]
        assert ranks == [1, 2, 3]


# ===========================================================================
#  Group 4: Integration — full pipeline
# ===========================================================================
class TestIntegration:
    """End-to-end: load model → split → score → select."""

    def test_full_pipeline(self, ce: CrossEncoder):
        """Mimics what the notebook will do."""
        # Split
        sentences = split_sentences(ABSTRACT)
        assert len(sentences) >= 3

        # Score
        scores = ce.score_query_vs_sentences(QUERY, sentences)
        assert len(scores) == len(sentences)

        # Select top-3
        top3 = select_top_sentences(QUERY, ABSTRACT, ce, top_n=3)
        assert len(top3) == 3
        assert top3[0]["score"] >= top3[1]["score"] >= top3[2]["score"]

        # Each selected sentence must actually come from the abstract
        for item in top3:
            assert item["sentence"] in sentences

    def test_relevant_vs_irrelevant_scoring(self, ce: CrossEncoder):
        """Cross-encoder should score relevant pairs higher than irrelevant ones."""
        relevant_score = ce.score_pairs([
            ("sleep apnea treatment", "CPAP is the gold standard for obstructive sleep apnea.")
        ])[0]
        irrelevant_score = ce.score_pairs([
            ("sleep apnea treatment", "The stock market rose 2% today.")
        ])[0]
        assert relevant_score > irrelevant_score, (
            f"Relevant ({relevant_score:.4f}) should > irrelevant ({irrelevant_score:.4f})"
        )


# ===========================================================================
#  Group 5: Notebook-cell logic replication
# ===========================================================================
class TestNotebookCellLogic:
    """Replicate key assertions from the notebook cells to catch issues early."""

    def test_smoke_test_cell(self, ce: CrossEncoder):
        """Cell 1.2: smoke test — score 1 pair, check float."""
        test_score = ce.score_query_vs_sentences(
            "sleep apnea", ["CPAP is effective."]
        )
        assert isinstance(test_score[0], float)

    def test_sentence_segmentation_cell(self, ce: CrossEncoder):
        """Cell 1.3: show sentences for an abstract."""
        abstract = ABSTRACT
        sentences = split_sentences(abstract)
        assert len(sentences) >= 3
        # Notebook prints them; we just confirm they are non-empty
        for s in sentences:
            assert len(s) > 0

    def test_sentence_scoring_demo_cell(self, ce: CrossEncoder):
        """Cell 1.4: select_top_sentences on a query + abstract."""
        top3 = select_top_sentences(QUERY, ABSTRACT, ce, top_n=3)
        assert len(top3) == 3
        # Scores are descending
        assert top3[0]["score"] >= top3[2]["score"]
        # Rank is 1-based sequential
        assert [t["rank"] for t in top3] == [1, 2, 3]

    def test_score_distribution_cell(self, ce: CrossEncoder):
        """Cell 1.5: collect scores for multiple abstracts (mimicked)."""
        # Simulate scoring sentences from 3 different abstracts
        abstracts = [
            "CPAP reduces OSA symptoms. Adherence is key for outcomes.",
            "Heart failure is a leading cause of death. Diuretics are used.",
            "The weather is sunny today. Birds are singing in the park.",
        ]
        all_scores = []
        for abs_text in abstracts:
            sents = split_sentences(abs_text)
            scores = ce.score_query_vs_sentences(QUERY, sents)
            all_scores.extend(scores)
        assert len(all_scores) >= 6  # at least 2 sentences per abstract
        assert all(isinstance(s, float) for s in all_scores)

    def test_ablation_random_vs_ce_cell(self, ce: CrossEncoder):
        """Cell 1.6: random baseline comparison (logic check)."""
        import random

        sentences = split_sentences(ABSTRACT)
        # CE selection
        top3_ce = select_top_sentences(QUERY, ABSTRACT, ce, top_n=3)
        ce_sents = {t["sentence"] for t in top3_ce}

        # Random selection
        random.seed(42)
        top3_rand = random.sample(sentences, min(3, len(sentences)))
        rand_sents = set(top3_rand)

        # Both should return 3 items
        assert len(ce_sents) == 3
        assert len(rand_sents) == 3
        # They CAN overlap, but the test just checks structure works
