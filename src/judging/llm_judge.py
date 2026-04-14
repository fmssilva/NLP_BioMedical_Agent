"""LLM-as-a-Judge evaluation for the RAG pipeline.

Provides prompt templates and judge functions for two evaluation axes
defined in the TREC BioGen task:

1. **Sentence alignment** — is each selected reference sentence relevant
   to the query?  Labels: Required / Unnecessary / Borderline / Inappropriate.

2. **Answer entailment** — is the generated answer fully supported by
   its cited reference sentences?  Labels: Supported / Partially Supported /
   Unsupported.

All judge functions accept a generic ``client`` that exposes
``client.chat.completions.create()`` (IAeduClient or openai.OpenAI).
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ── Valid label sets ────────────────────────────────────────────────────────
ALIGNMENT_LABELS = {"Required", "Unnecessary", "Borderline", "Inappropriate"}
ENTAILMENT_LABELS = {"Supported", "Partially Supported", "Unsupported"}

# ── Sentence Alignment prompts ─────────────────────────────────────────────

SENTENCE_ALIGNMENT_SYSTEM_PROMPT: str = (
    "You are a biomedical evaluation expert. Your task is to judge whether a "
    "reference sentence, cited in an answer to a biomedical question, is "
    "relevant and necessary for answering the question.\n\n"
    "Assign exactly ONE of the following labels:\n"
    "- Required: The sentence provides essential information for answering "
    "the question. Without it, the answer would be incomplete.\n"
    "- Unnecessary: The sentence is not needed — it is trivial, redundant, "
    "or not relevant to the specific question asked.\n"
    "- Borderline: The sentence is related to the topic and may be 'good "
    "to know', but is not strictly required for a complete answer.\n"
    "- Inappropriate: The sentence could mislead or harm the patient "
    "(e.g., contradictory advice, outdated information).\n\n"
    "Respond with a JSON object containing exactly one key \"label\" with "
    "the chosen label as its value. Do not include any other text."
)

SENTENCE_ALIGNMENT_USER_TEMPLATE: str = (
    "Question: {question}\n\n"
    "Reference sentence (PMID {pmid}):\n"
    "\"{sentence}\"\n\n"
    "Judge this sentence. Respond ONLY with JSON: {{\"label\": \"<label>\"}}"
)

# ── Answer Entailment prompts ──────────────────────────────────────────────

ENTAILMENT_SYSTEM_PROMPT: str = (
    "You are a biomedical evaluation expert. Your task is to judge whether a "
    "generated answer to a biomedical question is fully supported (entailed) "
    "by the provided reference sentences.\n\n"
    "Assign exactly ONE of the following labels:\n"
    "- Supported: The answer is completely supported by the reference "
    "sentences. Every claim in the answer can be traced back to the evidence.\n"
    "- Partially Supported: The answer is relevant and some claims are "
    "supported, but it also contains claims that go beyond what the "
    "reference sentences state.\n"
    "- Unsupported: The answer is not supported by the provided sentences "
    "and may contain fabricated information that could mislead or harm the "
    "patient.\n\n"
    "Respond with a JSON object containing:\n"
    "  \"label\": one of the three labels above\n"
    "  \"unsupported_claims\": a list of strings — each string is a claim "
    "from the answer that is NOT supported by the reference sentences. "
    "If the label is \"Supported\", this list should be empty.\n"
    "Do not include any other text."
)

ENTAILMENT_USER_TEMPLATE: str = (
    "Question: {question}\n\n"
    "Reference sentences:\n{references}\n\n"
    "Generated answer:\n\"{answer}\"\n\n"
    "Judge this answer. Respond ONLY with JSON: "
    "{{\"label\": \"<label>\", \"unsupported_claims\": [...]}}"
)

# ── Helpers ─────────────────────────────────────────────────────────────────

# Regex to find the first JSON object in a response string
_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _extract_json(text: str) -> Dict[str, Any]:
    """Best-effort extraction of a JSON object from an LLM response.

    The IAedu API does not support ``response_format`` so we parse from
    free-form text.
    """
    text = text.strip()
    # Try full text first (ideal case — model returned pure JSON)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: find first {...} block
    m = _JSON_OBJECT_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return {}


def _normalise_alignment_label(raw_label: str) -> str:
    """Map a raw label string to one of the canonical alignment labels."""
    raw = raw_label.strip().lower()
    for canonical in ALIGNMENT_LABELS:
        if canonical.lower() == raw:
            return canonical
    # Partial match fallback
    for canonical in ALIGNMENT_LABELS:
        if canonical.lower() in raw:
            return canonical
    return raw_label.strip()  # return as-is if unrecognised


def _normalise_entailment_label(raw_label: str) -> str:
    """Map a raw label string to one of the canonical entailment labels."""
    raw = raw_label.strip().lower()
    for canonical in ENTAILMENT_LABELS:
        if canonical.lower() == raw:
            return canonical
    # Partial match
    for canonical in ENTAILMENT_LABELS:
        if canonical.lower() in raw:
            return canonical
    return raw_label.strip()


def _llm_call_with_retry(
    client: Any,
    model_name: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 256,
    max_retries: int = 8,
    backoff_base: float = 2.0,
) -> str:
    """Call the LLM with exponential-backoff retry on rate-limit and connection errors.
    
    With 8 retries and base 2.0, the max cumulative wait is
    1+2+4+8+16+32+64+128 = 255 s (~4 min), enough for IAedu rate resets.
    """
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except RuntimeError as exc:
            last_err = exc
            if "429" in str(exc) or "rate limit" in str(exc).lower():
                wait = min(backoff_base ** attempt, 120)  # cap at 2 min
                logger.warning(
                    "Rate limit (attempt %d/%d), retrying in %.1fs …",
                    attempt + 1, max_retries, wait,
                )
                time.sleep(wait)
            else:
                raise
        except (ConnectionError, OSError, Exception) as exc:
            # Catch connection drops (ChunkedEncodingError, ProtocolError, etc.)
            err_name = type(exc).__name__
            if any(kw in err_name.lower() for kw in
                   ("chunked", "protocol", "connection", "timeout", "reset",
                    "encoding")):
                last_err = exc
                wait = min(backoff_base ** attempt, 120)
                logger.warning(
                    "Connection error %s (attempt %d/%d), retrying in %.1fs …",
                    err_name, attempt + 1, max_retries, wait,
                )
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(
        f"Judge call failed after {max_retries} retries: {last_err}"
    ) from last_err


# ── Public judge functions ──────────────────────────────────────────────────

def judge_sentence_alignment(
    question: str,
    sentence: str,
    pmid: str,
    client: Any,
    model_name: str = "gpt-4o",
    temperature: float = 0.0,
    max_retries: int = 5,
    backoff_base: float = 2.0,
) -> Dict[str, str]:
    """Judge whether a reference sentence is relevant to the question.

    Parameters
    ----------
    question : str
        The biomedical question.
    sentence : str
        The reference sentence to judge.
    pmid : str
        The PMID of the source document.
    client : IAeduClient | openai.OpenAI
        LLM client with ``.chat.completions.create()``.
    model_name : str
        Model identifier.
    temperature : float
        Sampling temperature (0 for deterministic judging).
    max_retries / backoff_base :
        Retry parameters for rate-limit handling.

    Returns
    -------
    dict
        ``{"label": str, "pmid": str}`` where label is one of
        Required / Unnecessary / Borderline / Inappropriate.
    """
    user_msg = SENTENCE_ALIGNMENT_USER_TEMPLATE.format(
        question=question,
        sentence=sentence,
        pmid=pmid,
    )
    messages = [
        {"role": "system", "content": SENTENCE_ALIGNMENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    raw = _llm_call_with_retry(
        client, model_name, messages,
        temperature=temperature,
        max_retries=max_retries,
        backoff_base=backoff_base,
    )

    parsed = _extract_json(raw)
    label = _normalise_alignment_label(parsed.get("label", raw))

    return {"label": label, "pmid": pmid}


def judge_answer_entailment(
    question: str,
    answer: str,
    reference_sentences: List[str],
    client: Any,
    model_name: str = "gpt-4o",
    temperature: float = 0.0,
    max_retries: int = 5,
    backoff_base: float = 2.0,
) -> Dict[str, Any]:
    """Judge whether a generated answer is entailed by its reference sentences.

    Parameters
    ----------
    question : str
        The biomedical question.
    answer : str
        The generated answer text.
    reference_sentences : list[str]
        The reference sentences used to generate the answer.
    client : IAeduClient | openai.OpenAI
        LLM client.
    model_name : str
        Model identifier.
    temperature : float
        Sampling temperature.
    max_retries / backoff_base :
        Retry parameters.

    Returns
    -------
    dict
        ``{"label": str, "unsupported_claims": list[str]}`` where label is
        one of Supported / Partially Supported / Unsupported.
    """
    refs_block = "\n".join(
        f"  {i + 1}. {s}" for i, s in enumerate(reference_sentences)
    )
    user_msg = ENTAILMENT_USER_TEMPLATE.format(
        question=question,
        answer=answer,
        references=refs_block,
    )
    messages = [
        {"role": "system", "content": ENTAILMENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    raw = _llm_call_with_retry(
        client, model_name, messages,
        temperature=temperature,
        max_retries=max_retries,
        backoff_base=backoff_base,
    )

    parsed = _extract_json(raw)
    label = _normalise_entailment_label(parsed.get("label", raw))
    unsupported = parsed.get("unsupported_claims", [])
    if not isinstance(unsupported, list):
        unsupported = [str(unsupported)] if unsupported else []

    return {"label": label, "unsupported_claims": unsupported}


def batch_judge_alignment(
    question: str,
    selected_sentences: List[Dict[str, Any]],
    client: Any,
    model_name: str = "gpt-4o",
    temperature: float = 0.0,
    inter_call_delay: float = 1.0,
    max_retries: int = 5,
    backoff_base: float = 2.0,
) -> List[Dict[str, str]]:
    """Judge sentence alignment for a batch of selected sentences.

    Parameters
    ----------
    question : str
        The biomedical question.
    selected_sentences : list[dict]
        Each dict must have ``"sentence"`` and ``"pmid"`` keys.
    client : IAeduClient | openai.OpenAI
        LLM client.
    model_name : str
        Model identifier.
    temperature : float
        Sampling temperature.
    inter_call_delay : float
        Seconds to wait between API calls to avoid rate limits.
    max_retries / backoff_base :
        Retry parameters.

    Returns
    -------
    list[dict]
        One ``{"label": str, "pmid": str}`` per input sentence.
    """
    results: List[Dict[str, str]] = []
    for i, entry in enumerate(selected_sentences):
        result = judge_sentence_alignment(
            question=question,
            sentence=entry["sentence"],
            pmid=entry["pmid"],
            client=client,
            model_name=model_name,
            temperature=temperature,
            max_retries=max_retries,
            backoff_base=backoff_base,
        )
        results.append(result)
        # Small delay between calls (except after last)
        if inter_call_delay > 0 and i < len(selected_sentences) - 1:
            time.sleep(inter_call_delay)
    return results
