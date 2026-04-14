"""
Comprehensive tests for ``src.judging``.

Tests cover:
  Group 1  judge_sentence_alignment — schema, label normalisation, retry, edge cases
  Group 2  judge_answer_entailment — schema, label normalisation, unsupported_claims, edge cases
  Group 3  batch_judge_alignment — length, schema, ordering, empty input
  Group 4  helper functions — _extract_json, _normalise_alignment_label, _normalise_entailment_label
  Group 5  prompt constants — non-empty, contain expected keywords

Run:
    python -m pytest src/judging/__judging_test.py -v --tb=short
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List
from unittest import mock

import pytest

# ── Ensure project root on sys.path ────────────────────────────────────────
import sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Imports under test ──────────────────────────────────────────────────────
from src.judging.llm_judge import (
    ALIGNMENT_LABELS,
    ENTAILMENT_LABELS,
    ENTAILMENT_SYSTEM_PROMPT,
    ENTAILMENT_USER_TEMPLATE,
    SENTENCE_ALIGNMENT_SYSTEM_PROMPT,
    SENTENCE_ALIGNMENT_USER_TEMPLATE,
    _extract_json,
    _normalise_alignment_label,
    _normalise_entailment_label,
    batch_judge_alignment,
    judge_answer_entailment,
    judge_sentence_alignment,
)


# ── Shared mock helpers ────────────────────────────────────────────────────

@dataclass
class _MockMessage:
    content: str = ""


@dataclass
class _MockChoice:
    message: _MockMessage = field(default_factory=_MockMessage)


@dataclass
class _MockCompletion:
    choices: List[_MockChoice] = field(default_factory=list)


def _make_mock_client(response_text: str) -> mock.MagicMock:
    """Return a mock client whose .chat.completions.create() returns the
    given text as the assistant message content."""
    completion = _MockCompletion(
        choices=[_MockChoice(message=_MockMessage(content=response_text))]
    )
    client = mock.MagicMock()
    client.chat.completions.create.return_value = completion
    return client


def _make_failing_then_ok_client(
    fail_count: int, response_text: str
) -> mock.MagicMock:
    """Return a mock client that raises RuntimeError('429') ``fail_count``
    times before returning ``response_text``."""
    completion = _MockCompletion(
        choices=[_MockChoice(message=_MockMessage(content=response_text))]
    )
    call_counter = {"n": 0}

    def side_effect(**kwargs):
        call_counter["n"] += 1
        if call_counter["n"] <= fail_count:
            raise RuntimeError("429 Too Many Requests")
        return completion

    client = mock.MagicMock()
    client.chat.completions.create.side_effect = side_effect
    return client


# ===========================================================================
# Group 1: judge_sentence_alignment
# ===========================================================================

class TestJudgeSentenceAlignmentSchema:
    """Verify output schema of judge_sentence_alignment."""

    def test_returns_dict(self):
        client = _make_mock_client('{"label": "Required"}')
        result = judge_sentence_alignment("What is aspirin?", "Aspirin is an NSAID.", "123", client)
        assert isinstance(result, dict)

    def test_has_label_key(self):
        client = _make_mock_client('{"label": "Required"}')
        result = judge_sentence_alignment("Q?", "Sentence.", "111", client)
        assert "label" in result

    def test_has_pmid_key(self):
        client = _make_mock_client('{"label": "Unnecessary"}')
        result = judge_sentence_alignment("Q?", "Sentence.", "222", client)
        assert "pmid" in result
        assert result["pmid"] == "222"

    def test_label_required(self):
        client = _make_mock_client('{"label": "Required"}')
        result = judge_sentence_alignment("Q?", "S.", "1", client)
        assert result["label"] == "Required"

    def test_label_unnecessary(self):
        client = _make_mock_client('{"label": "Unnecessary"}')
        result = judge_sentence_alignment("Q?", "S.", "1", client)
        assert result["label"] == "Unnecessary"

    def test_label_borderline(self):
        client = _make_mock_client('{"label": "Borderline"}')
        result = judge_sentence_alignment("Q?", "S.", "1", client)
        assert result["label"] == "Borderline"

    def test_label_inappropriate(self):
        client = _make_mock_client('{"label": "Inappropriate"}')
        result = judge_sentence_alignment("Q?", "S.", "1", client)
        assert result["label"] == "Inappropriate"

    def test_all_valid_labels(self):
        """All four canonical labels parse correctly."""
        for lbl in ALIGNMENT_LABELS:
            client = _make_mock_client(json.dumps({"label": lbl}))
            result = judge_sentence_alignment("Q?", "S.", "1", client)
            assert result["label"] == lbl


class TestJudgeSentenceAlignmentNormalisation:
    """Label normalisation from non-ideal LLM responses."""

    def test_lowercase_label(self):
        client = _make_mock_client('{"label": "required"}')
        result = judge_sentence_alignment("Q?", "S.", "1", client)
        assert result["label"] == "Required"

    def test_uppercase_label(self):
        client = _make_mock_client('{"label": "UNNECESSARY"}')
        result = judge_sentence_alignment("Q?", "S.", "1", client)
        assert result["label"] == "Unnecessary"

    def test_whitespace_around_label(self):
        client = _make_mock_client('{"label": "  Borderline  "}')
        result = judge_sentence_alignment("Q?", "S.", "1", client)
        assert result["label"] == "Borderline"

    def test_json_embedded_in_text(self):
        """Model wraps JSON in explanation text."""
        resp = 'Based on my analysis, the label is: {"label": "Required"} because ...'
        client = _make_mock_client(resp)
        result = judge_sentence_alignment("Q?", "S.", "1", client)
        assert result["label"] == "Required"

    def test_label_with_extra_text_partial_match(self):
        """Label text contains partial match."""
        client = _make_mock_client('{"label": "This is Inappropriate content"}')
        result = judge_sentence_alignment("Q?", "S.", "1", client)
        assert result["label"] == "Inappropriate"


class TestJudgeSentenceAlignmentRetry:
    """Retry logic on rate-limit errors."""

    @mock.patch("src.judging.llm_judge.time.sleep")
    def test_retries_on_429(self, mock_sleep):
        client = _make_failing_then_ok_client(2, '{"label": "Required"}')
        result = judge_sentence_alignment(
            "Q?", "S.", "1", client, max_retries=5, backoff_base=0.01
        )
        assert result["label"] == "Required"
        assert client.chat.completions.create.call_count == 3

    @mock.patch("src.judging.llm_judge.time.sleep")
    def test_gives_up_after_max_retries(self, mock_sleep):
        client = _make_failing_then_ok_client(10, '{"label": "Required"}')
        with pytest.raises(RuntimeError, match="failed after"):
            judge_sentence_alignment(
                "Q?", "S.", "1", client, max_retries=3, backoff_base=0.01
            )

    def test_non_429_error_propagates_immediately(self):
        client = mock.MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("Connection refused")
        with pytest.raises(RuntimeError, match="Connection refused"):
            judge_sentence_alignment("Q?", "S.", "1", client)


class TestJudgeSentenceAlignmentMessages:
    """Verify the messages sent to the LLM."""

    def test_system_message_included(self):
        client = _make_mock_client('{"label": "Required"}')
        judge_sentence_alignment("Q?", "S.", "1", client)
        call_kwargs = client.chat.completions.create.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        sys_msgs = [m for m in messages if m["role"] == "system"]
        assert len(sys_msgs) == 1
        assert "Required" in sys_msgs[0]["content"]

    def test_user_message_contains_question_and_sentence(self):
        client = _make_mock_client('{"label": "Required"}')
        judge_sentence_alignment("What is aspirin?", "Aspirin is an NSAID.", "999", client)
        call_kwargs = client.chat.completions.create.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        user_msgs = [m for m in messages if m["role"] == "user"]
        assert len(user_msgs) == 1
        assert "What is aspirin?" in user_msgs[0]["content"]
        assert "Aspirin is an NSAID." in user_msgs[0]["content"]
        assert "999" in user_msgs[0]["content"]

    def test_temperature_zero_by_default(self):
        client = _make_mock_client('{"label": "Required"}')
        judge_sentence_alignment("Q?", "S.", "1", client)
        call_kwargs = client.chat.completions.create.call_args
        assert call_kwargs.kwargs.get("temperature") == 0.0


# ===========================================================================
# Group 2: judge_answer_entailment
# ===========================================================================

class TestJudgeEntailmentSchema:
    """Verify output schema of judge_answer_entailment."""

    def test_returns_dict(self):
        client = _make_mock_client('{"label": "Supported", "unsupported_claims": []}')
        result = judge_answer_entailment("Q?", "Answer.", ["Ref."], client)
        assert isinstance(result, dict)

    def test_has_label_key(self):
        client = _make_mock_client('{"label": "Supported", "unsupported_claims": []}')
        result = judge_answer_entailment("Q?", "Answer.", ["Ref."], client)
        assert "label" in result

    def test_has_unsupported_claims_key(self):
        client = _make_mock_client('{"label": "Supported", "unsupported_claims": []}')
        result = judge_answer_entailment("Q?", "Answer.", ["Ref."], client)
        assert "unsupported_claims" in result

    def test_label_supported(self):
        client = _make_mock_client('{"label": "Supported", "unsupported_claims": []}')
        result = judge_answer_entailment("Q?", "A.", ["Ref."], client)
        assert result["label"] == "Supported"

    def test_label_partially_supported(self):
        client = _make_mock_client(
            '{"label": "Partially Supported", "unsupported_claims": ["claim X"]}'
        )
        result = judge_answer_entailment("Q?", "A.", ["Ref."], client)
        assert result["label"] == "Partially Supported"

    def test_label_unsupported(self):
        client = _make_mock_client(
            '{"label": "Unsupported", "unsupported_claims": ["all of it"]}'
        )
        result = judge_answer_entailment("Q?", "A.", ["Ref."], client)
        assert result["label"] == "Unsupported"

    def test_all_valid_labels(self):
        for lbl in ENTAILMENT_LABELS:
            resp = json.dumps({"label": lbl, "unsupported_claims": []})
            client = _make_mock_client(resp)
            result = judge_answer_entailment("Q?", "A.", ["Ref."], client)
            assert result["label"] == lbl

    def test_unsupported_claims_is_list(self):
        client = _make_mock_client(
            '{"label": "Partially Supported", "unsupported_claims": ["claim1", "claim2"]}'
        )
        result = judge_answer_entailment("Q?", "A.", ["Ref."], client)
        assert isinstance(result["unsupported_claims"], list)
        assert len(result["unsupported_claims"]) == 2

    def test_empty_unsupported_claims_for_supported(self):
        client = _make_mock_client('{"label": "Supported", "unsupported_claims": []}')
        result = judge_answer_entailment("Q?", "A.", ["R."], client)
        assert result["unsupported_claims"] == []


class TestJudgeEntailmentNormalisation:
    """Label normalisation from non-ideal LLM responses."""

    def test_lowercase_label(self):
        client = _make_mock_client('{"label": "supported", "unsupported_claims": []}')
        result = judge_answer_entailment("Q?", "A.", ["R."], client)
        assert result["label"] == "Supported"

    def test_uppercase_label(self):
        client = _make_mock_client('{"label": "UNSUPPORTED", "unsupported_claims": ["x"]}')
        result = judge_answer_entailment("Q?", "A.", ["R."], client)
        assert result["label"] == "Unsupported"

    def test_partial_match_partially_supported(self):
        client = _make_mock_client(
            '{"label": "partially supported", "unsupported_claims": []}'
        )
        result = judge_answer_entailment("Q?", "A.", ["R."], client)
        assert result["label"] == "Partially Supported"

    def test_json_in_text(self):
        resp = 'Here is my verdict: {"label": "Unsupported", "unsupported_claims": ["bad claim"]} end.'
        client = _make_mock_client(resp)
        result = judge_answer_entailment("Q?", "A.", ["R."], client)
        assert result["label"] == "Unsupported"
        assert "bad claim" in result["unsupported_claims"]

    def test_unsupported_claims_not_list_coerced(self):
        """If model returns a string instead of list, it gets wrapped."""
        client = _make_mock_client(
            '{"label": "Unsupported", "unsupported_claims": "single claim"}'
        )
        result = judge_answer_entailment("Q?", "A.", ["R."], client)
        assert isinstance(result["unsupported_claims"], list)
        assert len(result["unsupported_claims"]) == 1

    def test_missing_unsupported_claims_key(self):
        """If model omits unsupported_claims, defaults to empty list."""
        client = _make_mock_client('{"label": "Supported"}')
        result = judge_answer_entailment("Q?", "A.", ["R."], client)
        assert result["unsupported_claims"] == []


class TestJudgeEntailmentRetry:
    """Retry logic on rate-limit errors."""

    @mock.patch("src.judging.llm_judge.time.sleep")
    def test_retries_on_429(self, mock_sleep):
        client = _make_failing_then_ok_client(
            2, '{"label": "Supported", "unsupported_claims": []}'
        )
        result = judge_answer_entailment(
            "Q?", "A.", ["R."], client, max_retries=5, backoff_base=0.01
        )
        assert result["label"] == "Supported"

    @mock.patch("src.judging.llm_judge.time.sleep")
    def test_gives_up_after_max_retries(self, mock_sleep):
        client = _make_failing_then_ok_client(10, "irrelevant")
        with pytest.raises(RuntimeError, match="failed after"):
            judge_answer_entailment(
                "Q?", "A.", ["R."], client, max_retries=3, backoff_base=0.01
            )


class TestJudgeEntailmentMessages:
    """Verify the messages sent to the LLM."""

    def test_system_message_included(self):
        client = _make_mock_client('{"label": "Supported", "unsupported_claims": []}')
        judge_answer_entailment("Q?", "Answer text.", ["Ref1.", "Ref2."], client)
        call_kwargs = client.chat.completions.create.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        sys_msgs = [m for m in messages if m["role"] == "system"]
        assert len(sys_msgs) == 1
        assert "Supported" in sys_msgs[0]["content"]

    def test_user_message_contains_question_answer_refs(self):
        client = _make_mock_client('{"label": "Supported", "unsupported_claims": []}')
        judge_answer_entailment(
            "Is aspirin safe?", "Aspirin is safe.", ["Study shows safety."], client
        )
        call_kwargs = client.chat.completions.create.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        user_msgs = [m for m in messages if m["role"] == "user"]
        assert len(user_msgs) == 1
        assert "Is aspirin safe?" in user_msgs[0]["content"]
        assert "Aspirin is safe." in user_msgs[0]["content"]
        assert "Study shows safety." in user_msgs[0]["content"]

    def test_multiple_refs_numbered(self):
        client = _make_mock_client('{"label": "Supported", "unsupported_claims": []}')
        judge_answer_entailment("Q?", "A.", ["First.", "Second.", "Third."], client)
        call_kwargs = client.chat.completions.create.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        user_msg = [m for m in messages if m["role"] == "user"][0]["content"]
        assert "1." in user_msg
        assert "2." in user_msg
        assert "3." in user_msg


# ===========================================================================
# Group 3: batch_judge_alignment
# ===========================================================================

class TestBatchJudgeAlignment:
    """Verify batch_judge_alignment wraps judge_sentence_alignment."""

    def test_returns_list(self):
        client = _make_mock_client('{"label": "Required"}')
        selected = [
            {"sentence": "S1.", "pmid": "111"},
            {"sentence": "S2.", "pmid": "222"},
        ]
        result = batch_judge_alignment("Q?", selected, client, inter_call_delay=0)
        assert isinstance(result, list)

    def test_length_matches_input(self):
        client = _make_mock_client('{"label": "Borderline"}')
        selected = [
            {"sentence": "S1.", "pmid": "111"},
            {"sentence": "S2.", "pmid": "222"},
            {"sentence": "S3.", "pmid": "333"},
        ]
        result = batch_judge_alignment("Q?", selected, client, inter_call_delay=0)
        assert len(result) == 3

    def test_each_item_has_correct_schema(self):
        client = _make_mock_client('{"label": "Required"}')
        selected = [
            {"sentence": "S1.", "pmid": "111"},
            {"sentence": "S2.", "pmid": "222"},
        ]
        result = batch_judge_alignment("Q?", selected, client, inter_call_delay=0)
        for item in result:
            assert "label" in item
            assert "pmid" in item

    def test_pmids_match_input_order(self):
        client = _make_mock_client('{"label": "Unnecessary"}')
        selected = [
            {"sentence": "S1.", "pmid": "AAA"},
            {"sentence": "S2.", "pmid": "BBB"},
            {"sentence": "S3.", "pmid": "CCC"},
        ]
        result = batch_judge_alignment("Q?", selected, client, inter_call_delay=0)
        assert [r["pmid"] for r in result] == ["AAA", "BBB", "CCC"]

    def test_empty_input_returns_empty(self):
        client = _make_mock_client('{"label": "Required"}')
        result = batch_judge_alignment("Q?", [], client, inter_call_delay=0)
        assert result == []

    def test_single_sentence(self):
        client = _make_mock_client('{"label": "Inappropriate"}')
        selected = [{"sentence": "Bad advice.", "pmid": "999"}]
        result = batch_judge_alignment("Q?", selected, client, inter_call_delay=0)
        assert len(result) == 1
        assert result[0]["label"] == "Inappropriate"
        assert result[0]["pmid"] == "999"

    def test_calls_client_n_times(self):
        client = _make_mock_client('{"label": "Required"}')
        selected = [
            {"sentence": "S1.", "pmid": "1"},
            {"sentence": "S2.", "pmid": "2"},
        ]
        batch_judge_alignment("Q?", selected, client, inter_call_delay=0)
        assert client.chat.completions.create.call_count == 2

    @mock.patch("src.judging.llm_judge.time.sleep")
    def test_inter_call_delay_used(self, mock_sleep):
        client = _make_mock_client('{"label": "Required"}')
        selected = [
            {"sentence": "S1.", "pmid": "1"},
            {"sentence": "S2.", "pmid": "2"},
            {"sentence": "S3.", "pmid": "3"},
        ]
        batch_judge_alignment("Q?", selected, client, inter_call_delay=1.5)
        # Sleep called between calls (n-1 times)
        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(1.5)


# ===========================================================================
# Group 4: helper functions
# ===========================================================================

class TestExtractJson:
    """_extract_json parses JSON from various response formats."""

    def test_pure_json(self):
        assert _extract_json('{"label": "Required"}') == {"label": "Required"}

    def test_json_in_text(self):
        text = 'Some text {"label": "Unnecessary"} more text'
        assert _extract_json(text) == {"label": "Unnecessary"}

    def test_json_with_whitespace(self):
        assert _extract_json('  {"label": "X"}  ') == {"label": "X"}

    def test_no_json_returns_empty(self):
        assert _extract_json("no json here") == {}

    def test_empty_string_returns_empty(self):
        assert _extract_json("") == {}

    def test_malformed_json_returns_empty(self):
        assert _extract_json("{label: bad}") == {}

    def test_array_not_extracted(self):
        """We only extract objects, not arrays."""
        result = _extract_json("[1, 2, 3]")
        # Falls through to regex which finds objects only — returns {}
        assert result == {} or isinstance(result, list)

    def test_json_with_list_value(self):
        """Our regex only finds simple {...} blocks without nested braces,
        but for the prompt responses, unsupported_claims is a flat list."""
        text = '{"label": "Supported", "unsupported_claims": []}'
        result = _extract_json(text)
        assert result["label"] == "Supported"


class TestNormaliseAlignmentLabel:
    """_normalise_alignment_label canonical mapping."""

    def test_exact_match(self):
        for lbl in ALIGNMENT_LABELS:
            assert _normalise_alignment_label(lbl) == lbl

    def test_lowercase(self):
        assert _normalise_alignment_label("required") == "Required"
        assert _normalise_alignment_label("unnecessary") == "Unnecessary"
        assert _normalise_alignment_label("borderline") == "Borderline"
        assert _normalise_alignment_label("inappropriate") == "Inappropriate"

    def test_unknown_returned_as_is(self):
        assert _normalise_alignment_label("Unknown") == "Unknown"

    def test_whitespace_stripped(self):
        assert _normalise_alignment_label("  Required  ") == "Required"


class TestNormaliseEntailmentLabel:
    """_normalise_entailment_label canonical mapping."""

    def test_exact_match(self):
        for lbl in ENTAILMENT_LABELS:
            assert _normalise_entailment_label(lbl) == lbl

    def test_lowercase(self):
        assert _normalise_entailment_label("supported") == "Supported"
        assert _normalise_entailment_label("partially supported") == "Partially Supported"
        assert _normalise_entailment_label("unsupported") == "Unsupported"

    def test_unknown_returned_as_is(self):
        assert _normalise_entailment_label("Gibberish") == "Gibberish"

    def test_whitespace_stripped(self):
        assert _normalise_entailment_label("  Unsupported  ") == "Unsupported"


# ===========================================================================
# Group 5: prompt constants
# ===========================================================================

class TestPromptConstants:
    """Validate prompt constants have expected structure."""

    def test_alignment_system_prompt_nonempty(self):
        assert len(SENTENCE_ALIGNMENT_SYSTEM_PROMPT) > 50

    def test_alignment_system_prompt_contains_labels(self):
        for lbl in ALIGNMENT_LABELS:
            assert lbl in SENTENCE_ALIGNMENT_SYSTEM_PROMPT

    def test_alignment_user_template_has_placeholders(self):
        assert "{question}" in SENTENCE_ALIGNMENT_USER_TEMPLATE
        assert "{sentence}" in SENTENCE_ALIGNMENT_USER_TEMPLATE
        assert "{pmid}" in SENTENCE_ALIGNMENT_USER_TEMPLATE

    def test_entailment_system_prompt_nonempty(self):
        assert len(ENTAILMENT_SYSTEM_PROMPT) > 50

    def test_entailment_system_prompt_contains_labels(self):
        for lbl in ENTAILMENT_LABELS:
            assert lbl in ENTAILMENT_SYSTEM_PROMPT

    def test_entailment_user_template_has_placeholders(self):
        assert "{question}" in ENTAILMENT_USER_TEMPLATE
        assert "{answer}" in ENTAILMENT_USER_TEMPLATE
        assert "{references}" in ENTAILMENT_USER_TEMPLATE

    def test_alignment_labels_set_size(self):
        assert len(ALIGNMENT_LABELS) == 4

    def test_entailment_labels_set_size(self):
        assert len(ENTAILMENT_LABELS) == 3

    def test_system_prompts_mention_json(self):
        """Both prompts instruct the model to respond with JSON."""
        assert "JSON" in SENTENCE_ALIGNMENT_SYSTEM_PROMPT or "json" in SENTENCE_ALIGNMENT_SYSTEM_PROMPT.lower()
        assert "JSON" in ENTAILMENT_SYSTEM_PROMPT or "json" in ENTAILMENT_SYSTEM_PROMPT.lower()
