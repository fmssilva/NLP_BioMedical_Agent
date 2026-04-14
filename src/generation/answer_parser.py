"""Answer parser for the RAG pipeline.

Parses a raw LLM answer string, extracts inline ``[PMID X]`` citations,
computes word count, and validates constraints.
"""

from __future__ import annotations

import re
from typing import Dict, List, Set

# Regex to match citation groups like [PMID 12345] or [PMID 12345, PMID 67890]
_CITATION_GROUP_RE = re.compile(r"\[PMID\s+[\d,\s]+(?:PMID\s+[\d,\s]+)*\]", re.IGNORECASE)
# Regex to extract individual PMIDs from a citation group
_PMID_RE = re.compile(r"PMID\s+(\d+)", re.IGNORECASE)


def _split_answer_sentences(text: str) -> List[str]:
    """Split answer text into sentences, keeping citation brackets attached.

    Uses a simple heuristic: split on period followed by whitespace or EOL,
    but NOT on periods inside ``[PMID ...]`` brackets.
    """
    # Split on sentence-ending patterns: ". " or ".\n" or end-of-string after "."
    # We first normalise by stripping and ensuring the text ends without trailing whitespace.
    text = text.strip()
    if not text:
        return []

    # Use regex to split: period + (space or newline or end) that is NOT inside brackets
    # Simple approach: split on ". " or ".\n" then re-attach the period
    parts = re.split(r"(?<=\.)\s+(?=[A-Z\[])", text)
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences


def _extract_pmids_from_sentence(sentence: str) -> List[str]:
    """Extract all PMIDs cited in a single answer sentence."""
    pmids: List[str] = []
    for match in _CITATION_GROUP_RE.finditer(sentence):
        group_text = match.group(0)
        for pmid_match in _PMID_RE.finditer(group_text):
            pmid = pmid_match.group(1)
            if pmid not in pmids:
                pmids.append(pmid)
    return pmids


def _strip_citations(text: str) -> str:
    """Remove all ``[PMID ...]`` citation groups from text."""
    return _CITATION_GROUP_RE.sub("", text).strip()


def parse_answer(raw: str, valid_pmids: Set[str]) -> Dict:
    """Parse a raw LLM answer string into a structured dict.

    Parameters
    ----------
    raw : str
        The raw answer text from the LLM (with inline ``[PMID X]`` citations).
    valid_pmids : set[str]
        The set of corpus PMIDs.

    Returns
    -------
    dict
        ``{
            "text":       str,       # answer with citations intact
            "sentences":  [{"text": str, "pmids": [str, ...]}],
            "word_count": int,
            "all_pmids":  [str],     # flat, deduplicated, order of appearance
            "violations": {
                "over_word_limit":        bool,
                "sentences_over_3_pmids": [int],   # sentence indices
                "invalid_pmids":          [str],
            }
        }``
    """
    raw = (raw or "").strip()

    if not raw:
        return {
            "text": "",
            "sentences": [],
            "word_count": 0,
            "all_pmids": [],
            "violations": {
                "over_word_limit": False,
                "sentences_over_3_pmids": [],
                "invalid_pmids": [],
            },
        }

    sentences_raw = _split_answer_sentences(raw)

    sentences: List[Dict] = []
    all_pmids_ordered: List[str] = []

    for sent_text in sentences_raw:
        pmids = _extract_pmids_from_sentence(sent_text)
        sentences.append({"text": sent_text, "pmids": pmids})
        for p in pmids:
            if p not in all_pmids_ordered:
                all_pmids_ordered.append(p)

    # Word count: count on the text WITH citations stripped for fair comparison
    clean_text = _strip_citations(raw)
    # Normalise whitespace for word counting
    word_count = len(clean_text.split())

    # --- Violations ---
    over_word_limit = word_count > 250

    sentences_over_3 = [
        i for i, s in enumerate(sentences) if len(s["pmids"]) > 3
    ]

    invalid_pmids = [
        p for p in all_pmids_ordered if p not in valid_pmids
    ]

    return {
        "text": raw,
        "sentences": sentences,
        "word_count": word_count,
        "all_pmids": all_pmids_ordered,
        "violations": {
            "over_word_limit": over_word_limit,
            "sentences_over_3_pmids": sentences_over_3,
            "invalid_pmids": invalid_pmids,
        },
    }


def check_constraints(parsed: Dict) -> bool:
    """Return ``True`` if no constraint violations found.

    Parameters
    ----------
    parsed : dict
        Output of :func:`parse_answer`.
    """
    v = parsed.get("violations", {})
    if v.get("over_word_limit", False):
        return False
    if v.get("sentences_over_3_pmids", []):
        return False
    if v.get("invalid_pmids", []):
        return False
    return True
