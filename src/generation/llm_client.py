"""LLM client helpers for Phase 2.

Provides:
- ``get_gpt4o_client()``  — IAedu GPT-4o (streaming agent-chat endpoint).
- ``get_vlm_client()``    — vLLM on amalia.novasearch.org (OpenAI-compatible).
- ``get_vlm_model_name()``— first model id from the vLLM server.

The IAedu API is **not** OpenAI-compatible — it uses a streaming FormData POST
endpoint that returns newline-delimited JSON objects.  ``IAeduClient`` wraps
this behind the same ``.chat.completions.create()`` interface used by the
``openai`` library so that all downstream code (``generate_answer``,
``judge_*``) can use a single calling convention.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lightweight data-classes that mirror the openai response shape
# ---------------------------------------------------------------------------

@dataclass
class _Message:
    role: str = "assistant"
    content: str = ""


@dataclass
class _Choice:
    index: int = 0
    message: _Message = field(default_factory=_Message)
    finish_reason: str = "stop"


@dataclass
class _Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class _ChatCompletion:
    """Minimal replica of ``openai.types.chat.ChatCompletion``."""
    id: str = ""
    object: str = "chat.completion"
    model: str = ""
    choices: List[_Choice] = field(default_factory=list)
    usage: _Usage = field(default_factory=_Usage)


# ---------------------------------------------------------------------------
# IAedu streaming adapter
# ---------------------------------------------------------------------------

class _IAeduCompletions:
    """Implements the ``.create()`` method expected by downstream code."""

    def __init__(self, url: str, api_key: str, channel_id: str, timeout: int = 120):
        self.url = url
        self.api_key = api_key
        self.channel_id = channel_id
        self.timeout = timeout

    # ----- public API (matches openai.chat.completions.create) -----
    def create(
        self,
        *,
        model: str = "gpt-4o",
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 400,
        **kwargs: Any,
    ) -> _ChatCompletion:
        """Send *messages* to the IAedu agent-chat endpoint and return a
        ``_ChatCompletion`` that looks like an ``openai`` response.

        The IAedu endpoint ignores ``model``, ``temperature``, and
        ``max_tokens`` — the model is fixed server-side — but we accept
        them for interface compatibility.
        """
        # Flatten the messages list into a single prompt string.
        # IAedu has a single ``message`` field, not a chat history.
        prompt_parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"[System instructions]\n{content}")
            else:
                prompt_parts.append(content)
        prompt = "\n\n".join(prompt_parts)

        thread_id = str(uuid.uuid4()).replace("-", "")[:20]

        data = {
            "channel_id": self.channel_id,
            "thread_id": thread_id,
            "user_info": "{}",
            "message": prompt,
        }

        resp = requests.post(
            self.url,
            headers={"x-api-key": self.api_key},
            data=data,
            stream=True,
            timeout=self.timeout,
        )
        resp.raise_for_status()

        # Parse streaming response — newline-delimited JSON objects
        # Types observed: "start", "token" (content tokens), "error", "done"
        full_text = ""
        run_id = ""
        for raw_chunk in resp.iter_lines(decode_unicode=True):
            raw_chunk = raw_chunk.strip()
            if not raw_chunk:
                continue
            try:
                obj = json.loads(raw_chunk)
            except json.JSONDecodeError:
                # Sometimes the API returns non-JSON lines; skip them.
                logger.debug("IAedu: non-JSON line: %s", raw_chunk[:120])
                continue

            etype = obj.get("type", "")
            run_id = obj.get("run_id", run_id)

            if etype == "token":
                full_text += obj.get("content", "")
            elif etype == "error":
                error_msg = obj.get("content", "unknown error")
                raise RuntimeError(f"IAedu API error: {error_msg}")
            # "start" and "done" are control frames — skip.

        return _ChatCompletion(
            id=run_id,
            model=model,
            choices=[_Choice(message=_Message(content=full_text.strip()))],
        )


class _IAeduChatNamespace:
    """Mimics ``client.chat.completions``."""
    def __init__(self, completions: _IAeduCompletions):
        self.completions = completions


class IAeduClient:
    """Drop-in replacement for ``openai.OpenAI`` that talks to the IAedu
    streaming agent-chat endpoint.

    Usage::

        client = IAeduClient(url=..., api_key=..., channel_id=...)
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )
        print(resp.choices[0].message.content)
    """

    def __init__(self, url: str, api_key: str, channel_id: str, timeout: int = 120):
        self._completions = _IAeduCompletions(url, api_key, channel_id, timeout)
        self.chat = _IAeduChatNamespace(self._completions)


# ---------------------------------------------------------------------------
# Public factory functions
# ---------------------------------------------------------------------------

def get_gpt4o_client(
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    channel_id: Optional[str] = None,
    timeout: int = 120,
) -> IAeduClient:
    """Return an :class:`IAeduClient` configured for GPT-4o via the IAedu API.

    Reads ``IAEDU_URL``, ``IAEDU_KEY``, and ``IAEDU_CHANNEL`` from
    environment variables when explicit arguments are not provided.
    """
    url = url or os.getenv("IAEDU_URL", "")
    api_key = api_key or os.getenv("IAEDU_KEY", "")
    channel_id = channel_id or os.getenv("IAEDU_CHANNEL", "")

    if not url:
        raise EnvironmentError(
            "IAEDU_URL not set.  Add it to .env or pass url= explicitly."
        )
    if not api_key:
        raise EnvironmentError(
            "IAEDU_KEY not set.  Add it to .env or pass api_key= explicitly."
        )
    if not channel_id:
        raise EnvironmentError(
            "IAEDU_CHANNEL not set.  Add it to .env or pass channel_id= explicitly."
        )

    return IAeduClient(url=url, api_key=api_key, channel_id=channel_id, timeout=timeout)


def get_vlm_client(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> "openai.OpenAI":
    """Return a standard ``openai.OpenAI`` client pointing at the vLLM server.

    Reads ``VLLM_URL`` and ``VLLM_KEY`` from environment when not supplied.
    """
    import openai

    base_url = base_url or os.getenv("VLLM_URL", "")
    api_key = api_key or os.getenv("VLLM_KEY", "")

    if not base_url:
        raise EnvironmentError("VLLM_URL not set.")
    if not api_key:
        raise EnvironmentError("VLLM_KEY not set.")

    return openai.OpenAI(base_url=base_url, api_key=api_key)


def get_vlm_model_name(client: "openai.OpenAI") -> str:
    """Return the first model id advertised by the vLLM server."""
    models = client.models.list()
    if not models.data:
        raise RuntimeError("vLLM server returned no models.")
    return models.data[0].id
