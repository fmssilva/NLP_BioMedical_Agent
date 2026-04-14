"""Context builder for the RAG pipeline.

Formats selected reference sentences into a labelled context string
that the LLM can use for grounded answer generation.

Format per line::

    [PMID 12345678] Sentence text here.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Set

logger = logging.getLogger(__name__)


def build_context(
    selected: List[Dict],
    valid_pmids: Set[str],
) -> str:
    """Build a formatted context string from selected reference sentences.

    Parameters
    ----------
    selected : list[dict]
        Each dict must contain ``"pmid"`` (str) and ``"sentence"`` (str).
        An optional ``"score"`` key is accepted but ignored here.
    valid_pmids : set[str]
        The set of corpus PMIDs.  A warning is logged for any PMID that
        is not in this set (the sentence is still included).

    Returns
    -------
    str
        Newline-separated ``[PMID X] sentence`` lines, or ``""`` if
        *selected* is empty.
    """
    if not selected:
        return ""

    lines: list[str] = []
    for entry in selected:
        pmid = str(entry.get("pmid", "")).strip()
        sentence = str(entry.get("sentence", "")).strip()

        if not pmid or not sentence:
            logger.warning("Skipping entry with missing pmid or sentence: %s", entry)
            continue

        if pmid not in valid_pmids:
            logger.warning("PMID %s is not in the valid corpus set.", pmid)

        # Ensure the sentence ends with a period for consistency.
        if sentence and not sentence.endswith("."):
            sentence += "."

        lines.append(f"[PMID {pmid}] {sentence}")

    return "\n".join(lines)
