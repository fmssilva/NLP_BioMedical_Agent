"""Answer generator for the RAG pipeline.

Provides the prompt templates and the single ``generate_answer`` function
that calls an LLM (GPT-4o via IAedu or vLLM) to produce a cited biomedical
answer.

Prompt constants are module-level so the notebook can import and display them
directly.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates (shown verbatim in the notebook §3.4)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = (
    "You are a biomedical expert. Answer the given clinical question using ONLY "
    "the provided reference sentences. Each sentence in your answer MUST end with "
    "citations in the format [PMID X] or [PMID X, PMID Y]. Do not add any "
    "information not present in the references. Total answer must be 250 words or "
    "fewer. Maximum 3 PMIDs per sentence."
)

USER_PROMPT_TEMPLATE: str = (
    "Reference evidence:\n"
    "{context}\n"
    "\n"
    "Question: {question}\n"
    "\n"
    "Write a concise biomedical answer (max 250 words). "
    "Cite each sentence with [PMID X]."
)


# ---------------------------------------------------------------------------
# Generation function
# ---------------------------------------------------------------------------

def generate_answer(
    question: str,
    context: str,
    client: Any,
    model_name: str,
    temperature: float = 0.1,
    max_tokens: int = 400,
    max_retries: int = 5,
    backoff_base: float = 2.0,
) -> str:
    """Generate a cited biomedical answer using the LLM.

    Parameters
    ----------
    question : str
        The biomedical question.
    context : str
        Formatted reference evidence (output of ``build_context``).
    client : openai.OpenAI | IAeduClient
        An LLM client exposing ``.chat.completions.create()``.
    model_name : str
        Model identifier (e.g. ``"gpt-4o"``).
    temperature : float
        Sampling temperature (low = more deterministic).
    max_tokens : int
        Maximum tokens in the LLM response.
    max_retries : int
        Number of retry attempts on rate-limit / transient errors.
    backoff_base : float
        Exponential backoff base in seconds.

    Returns
    -------
    str
        The raw answer text (not yet parsed).
    """
    user_message = USER_PROMPT_TEMPLATE.format(
        context=context,
        question=question,
    )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

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
            # Catch rate-limit (429) and transient errors
            if "429" in str(exc) or "rate limit" in str(exc).lower():
                wait = backoff_base ** attempt
                logger.warning(
                    "Rate limit hit (attempt %d/%d), retrying in %.1fs …",
                    attempt + 1, max_retries, wait,
                )
                time.sleep(wait)
            else:
                raise  # non-transient error — propagate immediately

    raise RuntimeError(
        f"Failed after {max_retries} retries: {last_err}"
    ) from last_err
