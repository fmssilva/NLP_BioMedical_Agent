"""
Comprehensive tests for ``src.generation``.

Tests cover:
  Group 1  build_context – normal, empty, invalid PMIDs, edge cases
  Group 2  IAeduClient – streaming parse, error handling, interface shape (mocked)
  Group 3  get_gpt4o_client – env-var loading, missing env errors
  Group 4  get_vlm_client – env-var loading, missing env errors
  Group 5  get_vlm_model_name – model listing (mocked)
  Group 6  Response data-classes – shape matches openai conventions
  Group 7  answer_parser – parse_answer on hand-crafted strings
  Group 8  answer_parser – constraint violations
  Group 9  answer_generator – prompt templates, generate_answer (mocked)
  Group 10 bulk pipeline – output schema, constraint stats, JSON round-trip

Run:
    python -m pytest src/generation/__generation_test.py -v --tb=short
"""

from __future__ import annotations

import json
import os
import logging
from unittest import mock

import pytest

# ── Ensure project root on sys.path ────────────────────────────────────────
import sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Imports under test ──────────────────────────────────────────────────────
from src.generation.context_builder import build_context
from src.generation.llm_client import (
    IAeduClient,
    _ChatCompletion,
    _Choice,
    _IAeduCompletions,
    _Message,
    _Usage,
    get_gpt4o_client,
    get_vlm_client,
    get_vlm_model_name,
)


# ===========================================================================
# Group 1: build_context
# ===========================================================================

class TestBuildContextBasic:
    """Core functionality of build_context."""

    VALID = {"111", "222", "333"}

    def test_single_entry(self):
        selected = [{"pmid": "111", "sentence": "Aspirin helps", "score": 5.0}]
        result = build_context(selected, self.VALID)
        assert result == "[PMID 111] Aspirin helps."

    def test_multiple_entries(self):
        selected = [
            {"pmid": "111", "sentence": "First.", "score": 5.0},
            {"pmid": "222", "sentence": "Second.", "score": 3.0},
        ]
        result = build_context(selected, self.VALID)
        lines = result.split("\n")
        assert len(lines) == 2
        assert lines[0] == "[PMID 111] First."
        assert lines[1] == "[PMID 222] Second."

    def test_adds_trailing_period(self):
        """Sentence without period gets one appended."""
        selected = [{"pmid": "111", "sentence": "No period here", "score": 1.0}]
        result = build_context(selected, self.VALID)
        assert result.endswith(".")

    def test_no_double_period(self):
        """Sentence already ending with period keeps just one."""
        selected = [{"pmid": "111", "sentence": "Already has one.", "score": 1.0}]
        result = build_context(selected, self.VALID)
        assert result.endswith("one.")
        assert not result.endswith("one..")

    def test_empty_list_returns_empty_string(self):
        assert build_context([], self.VALID) == ""

    def test_preserves_order(self):
        selected = [
            {"pmid": "333", "sentence": "Third.", "score": 1.0},
            {"pmid": "111", "sentence": "First.", "score": 5.0},
        ]
        result = build_context(selected, self.VALID)
        lines = result.split("\n")
        assert "333" in lines[0]
        assert "111" in lines[1]

    def test_score_key_optional(self):
        """Entry without 'score' key should work fine."""
        selected = [{"pmid": "111", "sentence": "Test."}]
        result = build_context(selected, self.VALID)
        assert "[PMID 111]" in result


class TestBuildContextInvalidPMIDs:
    """build_context behaviour with PMIDs not in the valid corpus set."""

    VALID = {"111"}

    def test_invalid_pmid_warns_but_includes(self, caplog):
        """Invalid PMID triggers a warning but the sentence is still included."""
        selected = [{"pmid": "999", "sentence": "Unknown ref.", "score": 1.0}]
        with caplog.at_level(logging.WARNING):
            result = build_context(selected, self.VALID)
        assert "[PMID 999]" in result
        assert "not in the valid corpus" in caplog.text

    def test_mix_valid_and_invalid(self, caplog):
        selected = [
            {"pmid": "111", "sentence": "Valid ref.", "score": 1.0},
            {"pmid": "BAD", "sentence": "Bad ref.", "score": 0.5},
        ]
        with caplog.at_level(logging.WARNING):
            result = build_context(selected, self.VALID)
        lines = result.split("\n")
        assert len(lines) == 2
        assert "BAD" in caplog.text


class TestBuildContextEdgeCases:
    """Edge cases for build_context."""

    VALID = {"111"}

    def test_missing_pmid_key_skips(self, caplog):
        selected = [{"sentence": "No pmid here.", "score": 1.0}]
        with caplog.at_level(logging.WARNING):
            result = build_context(selected, self.VALID)
        assert result == ""
        assert "missing pmid" in caplog.text.lower() or "Skipping" in caplog.text

    def test_missing_sentence_key_skips(self, caplog):
        selected = [{"pmid": "111"}]
        with caplog.at_level(logging.WARNING):
            result = build_context(selected, self.VALID)
        assert result == ""

    def test_empty_pmid_string_skips(self, caplog):
        selected = [{"pmid": "", "sentence": "Text.", "score": 1.0}]
        with caplog.at_level(logging.WARNING):
            result = build_context(selected, self.VALID)
        assert result == ""

    def test_empty_sentence_string_skips(self, caplog):
        selected = [{"pmid": "111", "sentence": "", "score": 1.0}]
        with caplog.at_level(logging.WARNING):
            result = build_context(selected, self.VALID)
        assert result == ""

    def test_whitespace_sentence_stripped(self):
        selected = [{"pmid": "111", "sentence": "  Some text  ", "score": 1.0}]
        result = build_context(selected, {"111"})
        assert result == "[PMID 111] Some text."

    def test_valid_pmids_empty_set(self, caplog):
        """All PMIDs warn when valid set is empty."""
        selected = [{"pmid": "111", "sentence": "Test.", "score": 1.0}]
        with caplog.at_level(logging.WARNING):
            result = build_context(selected, set())
        assert "[PMID 111]" in result
        assert "not in the valid corpus" in caplog.text


# ===========================================================================
# Group 2: IAeduClient — mocked streaming response
# ===========================================================================

def _make_streaming_response(chunks: list[dict], status_code: int = 200):
    """Create a mock ``requests.Response`` that yields newline-delimited JSON."""
    lines = [json.dumps(c) for c in chunks]
    body = "\n\n".join(lines) + "\n\n"

    resp = mock.MagicMock()
    resp.status_code = status_code
    resp.raise_for_status = mock.MagicMock()
    resp.iter_lines = mock.MagicMock(
        return_value=iter(body.split("\n"))
    )
    return resp


class TestIAeduClientCreate:
    """Test the IAedu streaming parse logic with mocked HTTP."""

    def _make_client(self) -> IAeduClient:
        return IAeduClient(
            url="https://fake.iaedu.pt/stream",
            api_key="sk-test",
            channel_id="ch-test",
        )

    @mock.patch("src.generation.llm_client.requests.post")
    def test_basic_token_assembly(self, mock_post):
        """Tokens are concatenated into the final message content."""
        chunks = [
            {"run_id": "r1", "type": "start", "content": "Processing"},
            {"run_id": "r1", "type": "token", "content": "Hello"},
            {"run_id": "r1", "type": "token", "content": " world"},
            {"run_id": "r1", "type": "done", "content": "r1", "messageId": "m1"},
        ]
        mock_post.return_value = _make_streaming_response(chunks)

        client = self._make_client()
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert resp.choices[0].message.content == "Hello world"
        assert resp.id == "r1"
        assert resp.model == "gpt-4o"

    @mock.patch("src.generation.llm_client.requests.post")
    def test_error_type_raises(self, mock_post):
        """An error frame from IAedu should raise RuntimeError."""
        chunks = [
            {"run_id": "r1", "type": "start", "content": "Processing"},
            {"run_id": "r1", "type": "error", "content": "Rate limit reached (429)"},
            {"run_id": "r1", "type": "done", "content": "r1"},
        ]
        mock_post.return_value = _make_streaming_response(chunks)

        client = self._make_client()
        with pytest.raises(RuntimeError, match="Rate limit"):
            client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hi"}],
            )

    @mock.patch("src.generation.llm_client.requests.post")
    def test_empty_tokens_returns_empty(self, mock_post):
        """If no token frames arrive, content is empty string."""
        chunks = [
            {"run_id": "r1", "type": "start", "content": "Processing"},
            {"run_id": "r1", "type": "done", "content": "r1"},
        ]
        mock_post.return_value = _make_streaming_response(chunks)

        client = self._make_client()
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert resp.choices[0].message.content == ""

    @mock.patch("src.generation.llm_client.requests.post")
    def test_system_and_user_messages_combined(self, mock_post):
        """System + user messages are flattened into the IAedu 'message' field."""
        chunks = [
            {"run_id": "r1", "type": "start", "content": "Processing"},
            {"run_id": "r1", "type": "token", "content": "OK"},
            {"run_id": "r1", "type": "done", "content": "r1"},
        ]
        mock_post.return_value = _make_streaming_response(chunks)

        client = self._make_client()
        client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
        )

        # Verify the data sent to requests.post
        call_kwargs = mock_post.call_args
        sent_data = call_kwargs.kwargs.get("data") or call_kwargs[1].get("data")
        assert "[System instructions]" in sent_data["message"]
        assert "Hello" in sent_data["message"]

    @mock.patch("src.generation.llm_client.requests.post")
    def test_response_has_openai_shape(self, mock_post):
        """Response matches the openai ChatCompletion interface."""
        chunks = [
            {"run_id": "r1", "type": "start", "content": "Processing"},
            {"run_id": "r1", "type": "token", "content": "Hi"},
            {"run_id": "r1", "type": "done", "content": "r1"},
        ]
        mock_post.return_value = _make_streaming_response(chunks)

        client = self._make_client()
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )

        # Check interface attributes match openai conventions
        assert hasattr(resp, "choices")
        assert hasattr(resp, "id")
        assert hasattr(resp, "model")
        assert hasattr(resp, "usage")
        assert hasattr(resp.choices[0], "message")
        assert hasattr(resp.choices[0].message, "content")
        assert hasattr(resp.choices[0].message, "role")
        assert resp.choices[0].message.role == "assistant"
        assert resp.choices[0].finish_reason == "stop"

    @mock.patch("src.generation.llm_client.requests.post")
    def test_non_json_lines_skipped(self, mock_post):
        """Non-JSON lines in the stream are silently ignored."""
        body = (
            "not json at all\n"
            + json.dumps({"run_id": "r1", "type": "start", "content": "go"}) + "\n"
            + "another bad line\n"
            + json.dumps({"run_id": "r1", "type": "token", "content": "OK"}) + "\n"
            + json.dumps({"run_id": "r1", "type": "done", "content": "r1"}) + "\n"
        )
        resp = mock.MagicMock()
        resp.status_code = 200
        resp.raise_for_status = mock.MagicMock()
        resp.iter_lines = mock.MagicMock(return_value=iter(body.split("\n")))
        mock_post.return_value = resp

        client = self._make_client()
        result = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert result.choices[0].message.content == "OK"

    @mock.patch("src.generation.llm_client.requests.post")
    def test_multiline_token_content(self, mock_post):
        """Token content can contain newlines."""
        chunks = [
            {"run_id": "r1", "type": "start", "content": "Processing"},
            {"run_id": "r1", "type": "token", "content": "Line1\nLine2"},
            {"run_id": "r1", "type": "token", "content": "\nLine3"},
            {"run_id": "r1", "type": "done", "content": "r1"},
        ]
        mock_post.return_value = _make_streaming_response(chunks)

        client = self._make_client()
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert resp.choices[0].message.content == "Line1\nLine2\nLine3"

    @mock.patch("src.generation.llm_client.requests.post")
    def test_content_is_stripped(self, mock_post):
        """Trailing/leading whitespace in assembled text is stripped."""
        chunks = [
            {"run_id": "r1", "type": "start", "content": "Processing"},
            {"run_id": "r1", "type": "token", "content": "  Hello  "},
            {"run_id": "r1", "type": "done", "content": "r1"},
        ]
        mock_post.return_value = _make_streaming_response(chunks)

        client = self._make_client()
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert resp.choices[0].message.content == "Hello"


# ===========================================================================
# Group 3: get_gpt4o_client — env var loading
# ===========================================================================

class TestGetGpt4oClient:
    """Tests for the get_gpt4o_client factory function."""

    def test_explicit_args(self):
        """Passing all args directly should not read env."""
        client = get_gpt4o_client(
            url="https://x.com/stream",
            api_key="sk-test",
            channel_id="ch-test",
        )
        assert isinstance(client, IAeduClient)

    def test_reads_env_vars(self):
        env = {
            "IAEDU_URL": "https://x.com/stream",
            "IAEDU_KEY": "sk-test",
            "IAEDU_CHANNEL": "ch-test",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            client = get_gpt4o_client()
        assert isinstance(client, IAeduClient)

    def test_missing_url_raises(self):
        env = {"IAEDU_URL": "", "IAEDU_KEY": "k", "IAEDU_CHANNEL": "c"}
        with mock.patch.dict(os.environ, env, clear=False):
            with pytest.raises(EnvironmentError, match="IAEDU_URL"):
                get_gpt4o_client()

    def test_missing_key_raises(self):
        env = {"IAEDU_URL": "u", "IAEDU_KEY": "", "IAEDU_CHANNEL": "c"}
        with mock.patch.dict(os.environ, env, clear=False):
            with pytest.raises(EnvironmentError, match="IAEDU_KEY"):
                get_gpt4o_client()

    def test_missing_channel_raises(self):
        env = {"IAEDU_URL": "u", "IAEDU_KEY": "k", "IAEDU_CHANNEL": ""}
        with mock.patch.dict(os.environ, env, clear=False):
            with pytest.raises(EnvironmentError, match="IAEDU_CHANNEL"):
                get_gpt4o_client()


# ===========================================================================
# Group 4: get_vlm_client — env var loading
# ===========================================================================

class TestGetVlmClient:
    """Tests for get_vlm_client factory (mocked openai.OpenAI)."""

    def test_explicit_args(self):
        mock_openai_mod = mock.MagicMock()
        with mock.patch.dict("sys.modules", {"openai": mock_openai_mod}):
            client = get_vlm_client(base_url="https://x.com/v1", api_key="key")
        mock_openai_mod.OpenAI.assert_called_once_with(base_url="https://x.com/v1", api_key="key")
        assert client is mock_openai_mod.OpenAI.return_value

    def test_reads_env(self):
        mock_openai_mod = mock.MagicMock()
        env = {"VLLM_URL": "https://v.com/v1", "VLLM_KEY": "vk"}
        with mock.patch.dict("sys.modules", {"openai": mock_openai_mod}), \
             mock.patch.dict(os.environ, env, clear=False):
            get_vlm_client()
        mock_openai_mod.OpenAI.assert_called_once_with(base_url="https://v.com/v1", api_key="vk")

    def test_missing_url_raises(self):
        env = {"VLLM_URL": "", "VLLM_KEY": "k"}
        with mock.patch.dict(os.environ, env, clear=False):
            with pytest.raises(EnvironmentError, match="VLLM_URL"):
                get_vlm_client()

    def test_missing_key_raises(self):
        env = {"VLLM_URL": "u", "VLLM_KEY": ""}
        with mock.patch.dict(os.environ, env, clear=False):
            with pytest.raises(EnvironmentError, match="VLLM_KEY"):
                get_vlm_client()


# ===========================================================================
# Group 5: get_vlm_model_name — mocked
# ===========================================================================

class TestGetVlmModelName:
    """Tests for get_vlm_model_name."""

    def test_returns_first_model_id(self):
        mock_client = mock.MagicMock()
        model1 = mock.MagicMock()
        model1.id = "meta-llama/Llama-3-70B"
        model2 = mock.MagicMock()
        model2.id = "other-model"
        mock_client.models.list.return_value.data = [model1, model2]

        name = get_vlm_model_name(mock_client)
        assert name == "meta-llama/Llama-3-70B"

    def test_empty_model_list_raises(self):
        mock_client = mock.MagicMock()
        mock_client.models.list.return_value.data = []

        with pytest.raises(RuntimeError, match="no models"):
            get_vlm_model_name(mock_client)


# ===========================================================================
# Group 6: Response data-classes
# ===========================================================================

class TestResponseDataclasses:
    """Verify the response data-classes have the correct defaults."""

    def test_message_defaults(self):
        m = _Message()
        assert m.role == "assistant"
        assert m.content == ""

    def test_choice_defaults(self):
        c = _Choice()
        assert c.index == 0
        assert c.finish_reason == "stop"
        assert isinstance(c.message, _Message)

    def test_usage_defaults(self):
        u = _Usage()
        assert u.prompt_tokens == 0

    def test_chat_completion_defaults(self):
        cc = _ChatCompletion()
        assert cc.object == "chat.completion"
        assert cc.choices == []
        assert isinstance(cc.usage, _Usage)

    def test_chat_completion_with_choices(self):
        cc = _ChatCompletion(
            id="test",
            model="gpt-4o",
            choices=[_Choice(message=_Message(content="hi"))],
        )
        assert cc.choices[0].message.content == "hi"


# ===========================================================================
# Group 7: answer_parser — parse_answer on hand-crafted strings
# ===========================================================================

from src.generation.answer_parser import parse_answer, check_constraints

# A clean, well-formed 2-sentence answer used in multiple tests.
_CLEAN_ANSWER = (
    "Aspirin inhibits platelet aggregation through COX-1 blockade [PMID 111]. "
    "Low-dose aspirin is recommended for secondary cardiovascular prevention [PMID 222]."
)
_VALID_PMIDS = {"111", "222", "333", "444"}


class TestParseAnswerBasic:
    """Group 7a: basic parsing of well-formed answers."""

    def test_two_sentence_answer(self):
        parsed = parse_answer(_CLEAN_ANSWER, _VALID_PMIDS)
        assert len(parsed["sentences"]) == 2
        assert parsed["sentences"][0]["pmids"] == ["111"]
        assert parsed["sentences"][1]["pmids"] == ["222"]

    def test_word_count_excludes_citations(self):
        parsed = parse_answer(_CLEAN_ANSWER, _VALID_PMIDS)
        # "Aspirin inhibits platelet aggregation through COX-1 blockade.
        #  Low-dose aspirin is recommended for secondary cardiovascular prevention."
        # That's 15 words (citations stripped).
        assert parsed["word_count"] > 0
        assert parsed["word_count"] < 250

    def test_all_pmids_deduplicated(self):
        answer = "A [PMID 111]. B [PMID 111, PMID 222]."
        parsed = parse_answer(answer, _VALID_PMIDS)
        assert parsed["all_pmids"] == ["111", "222"]

    def test_text_field_preserves_citations(self):
        parsed = parse_answer(_CLEAN_ANSWER, _VALID_PMIDS)
        assert "[PMID 111]" in parsed["text"]

    def test_empty_answer(self):
        parsed = parse_answer("", _VALID_PMIDS)
        assert parsed["text"] == ""
        assert parsed["sentences"] == []
        assert parsed["word_count"] == 0
        assert parsed["all_pmids"] == []
        assert check_constraints(parsed) is True

    def test_none_answer(self):
        parsed = parse_answer(None, _VALID_PMIDS)
        assert parsed["text"] == ""
        assert check_constraints(parsed) is True

    def test_answer_no_citations(self):
        """An answer without any [PMID] tags — no pmids extracted."""
        parsed = parse_answer("Some text without citations.", _VALID_PMIDS)
        assert parsed["all_pmids"] == []
        assert parsed["sentences"][0]["pmids"] == []

    def test_multi_pmid_per_sentence(self):
        answer = "Drug X and Y are both effective [PMID 111, PMID 222]."
        parsed = parse_answer(answer, _VALID_PMIDS)
        assert set(parsed["sentences"][0]["pmids"]) == {"111", "222"}

    def test_three_pmids_per_sentence_ok(self):
        answer = "Result here [PMID 111, PMID 222, PMID 333]."
        parsed = parse_answer(answer, _VALID_PMIDS)
        assert len(parsed["sentences"][0]["pmids"]) == 3
        assert parsed["violations"]["sentences_over_3_pmids"] == []

    def test_pmids_order_of_appearance(self):
        answer = "A [PMID 333]. B [PMID 111]. C [PMID 222]."
        parsed = parse_answer(answer, _VALID_PMIDS)
        assert parsed["all_pmids"] == ["333", "111", "222"]


# ===========================================================================
# Group 8: answer_parser — constraint violations
# ===========================================================================

class TestParseAnswerViolations:
    """Group 8: constraint violation detection."""

    def test_over_word_limit(self):
        # Build an answer >250 words
        long_answer = " ".join(["word"] * 260) + " [PMID 111]."
        parsed = parse_answer(long_answer, _VALID_PMIDS)
        assert parsed["violations"]["over_word_limit"] is True
        assert check_constraints(parsed) is False

    def test_under_word_limit(self):
        parsed = parse_answer(_CLEAN_ANSWER, _VALID_PMIDS)
        assert parsed["violations"]["over_word_limit"] is False
        assert check_constraints(parsed) is True

    def test_exactly_250_words(self):
        # "final[PMID 111]." → strip → "final." (one token) ⇒ 249 + 1 = 250
        answer = " ".join(["word"] * 249) + " final[PMID 111]."
        parsed = parse_answer(answer, _VALID_PMIDS)
        assert parsed["word_count"] == 250
        assert parsed["violations"]["over_word_limit"] is False

    def test_four_pmids_in_one_sentence(self):
        answer = "Result [PMID 111, PMID 222, PMID 333, PMID 444]."
        parsed = parse_answer(answer, _VALID_PMIDS)
        assert 0 in parsed["violations"]["sentences_over_3_pmids"]
        assert check_constraints(parsed) is False

    def test_invalid_pmid(self):
        answer = "Result [PMID 999]."
        parsed = parse_answer(answer, _VALID_PMIDS)
        assert "999" in parsed["violations"]["invalid_pmids"]
        assert check_constraints(parsed) is False

    def test_mix_valid_and_invalid_pmids(self):
        answer = "A [PMID 111]. B [PMID 888]."
        parsed = parse_answer(answer, _VALID_PMIDS)
        assert "888" in parsed["violations"]["invalid_pmids"]
        assert "111" not in parsed["violations"]["invalid_pmids"]

    def test_clean_answer_passes(self):
        parsed = parse_answer(_CLEAN_ANSWER, _VALID_PMIDS)
        assert check_constraints(parsed) is True

    def test_multiple_violations_at_once(self):
        long_text = " ".join(["word"] * 260)
        answer = f"{long_text} [PMID 111, PMID 222, PMID 333, PMID 444, PMID 999]."
        parsed = parse_answer(answer, _VALID_PMIDS)
        assert parsed["violations"]["over_word_limit"] is True
        assert len(parsed["violations"]["sentences_over_3_pmids"]) > 0
        assert "999" in parsed["violations"]["invalid_pmids"]
        assert check_constraints(parsed) is False


class TestCheckConstraints:
    """Group 8b: check_constraints edge cases."""

    def test_empty_violations_dict(self):
        """Manually constructed parsed dict with no violations."""
        parsed = {
            "violations": {
                "over_word_limit": False,
                "sentences_over_3_pmids": [],
                "invalid_pmids": [],
            }
        }
        assert check_constraints(parsed) is True

    def test_missing_violations_key(self):
        """If violations key missing, should still return True (no violations)."""
        assert check_constraints({}) is True


# ===========================================================================
# Group 9: answer_generator — prompt templates and generate_answer (mocked)
# ===========================================================================

from src.generation.answer_generator import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    generate_answer,
)


class TestPromptTemplates:
    """Group 9a: prompt template content checks."""

    def test_system_prompt_mentions_250_words(self):
        assert "250" in SYSTEM_PROMPT

    def test_system_prompt_mentions_pmid(self):
        assert "PMID" in SYSTEM_PROMPT

    def test_system_prompt_mentions_3_pmids(self):
        assert "3" in SYSTEM_PROMPT

    def test_user_template_has_context_placeholder(self):
        assert "{context}" in USER_PROMPT_TEMPLATE

    def test_user_template_has_question_placeholder(self):
        assert "{question}" in USER_PROMPT_TEMPLATE

    def test_user_template_formats_correctly(self):
        filled = USER_PROMPT_TEMPLATE.format(
            context="[PMID 1] Test sentence.",
            question="What is X?",
        )
        assert "[PMID 1] Test sentence." in filled
        assert "What is X?" in filled


class TestGenerateAnswer:
    """Group 9b: generate_answer with mocked client."""

    def test_returns_content_string(self):
        mock_client = mock.MagicMock()
        mock_client.chat.completions.create.return_value = mock.MagicMock(
            choices=[mock.MagicMock(message=mock.MagicMock(content="Answer [PMID 1]."))]
        )
        result = generate_answer(
            question="What is aspirin?",
            context="[PMID 1] Aspirin is a drug.",
            client=mock_client,
            model_name="gpt-4o",
        )
        assert result == "Answer [PMID 1]."

    def test_passes_system_and_user_messages(self):
        mock_client = mock.MagicMock()
        mock_client.chat.completions.create.return_value = mock.MagicMock(
            choices=[mock.MagicMock(message=mock.MagicMock(content="OK"))]
        )
        generate_answer("Q?", "ctx", mock_client, "gpt-4o")
        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "Q?" in messages[1]["content"]
        assert "ctx" in messages[1]["content"]

    def test_respects_temperature_and_max_tokens(self):
        mock_client = mock.MagicMock()
        mock_client.chat.completions.create.return_value = mock.MagicMock(
            choices=[mock.MagicMock(message=mock.MagicMock(content="OK"))]
        )
        generate_answer("Q?", "ctx", mock_client, "m", temperature=0.5, max_tokens=100)
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 100

    def test_default_temperature_is_low(self):
        mock_client = mock.MagicMock()
        mock_client.chat.completions.create.return_value = mock.MagicMock(
            choices=[mock.MagicMock(message=mock.MagicMock(content="OK"))]
        )
        generate_answer("Q?", "ctx", mock_client, "m")
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.1


# ════════════════════════════════════════════════════════════════════════════
# Group 10: Bulk pipeline — output schema, constraint stats, JSON round-trip
# ════════════════════════════════════════════════════════════════════════════


def _make_bulk_entry(qid: str, question: str, raw: str, valid_pmids: set) -> dict:
    """Simulate what run_bulk_rag_pipeline produces for one query."""
    parsed = parse_answer(raw, valid_pmids)
    return {
        "query_id": qid,
        "question": question,
        "selected_sentences": [
            {"pmid": "111", "sentence": "Test sentence.", "score": 1.0, "rank": 0}
        ],
        "raw_answer": raw,
        "parsed": parsed,
    }


class TestBulkPipelineSchema:
    """Group 10a: output schema from run_bulk_rag_pipeline."""

    def test_entry_has_required_keys(self):
        entry = _make_bulk_entry("1", "Q?", "Answer [PMID 111].", _VALID_PMIDS)
        for key in ("query_id", "question", "selected_sentences", "raw_answer", "parsed"):
            assert key in entry, f"Missing key: {key}"

    def test_parsed_sub_dict_has_required_keys(self):
        entry = _make_bulk_entry("1", "Q?", "Answer [PMID 111].", _VALID_PMIDS)
        parsed = entry["parsed"]
        for key in ("text", "sentences", "word_count", "all_pmids", "violations"):
            assert key in parsed, f"Missing parsed key: {key}"

    def test_query_id_is_string(self):
        entry = _make_bulk_entry("42", "Q?", "Answer [PMID 111].", _VALID_PMIDS)
        assert isinstance(entry["query_id"], str)

    def test_selected_sentences_list_of_dicts(self):
        entry = _make_bulk_entry("1", "Q?", "Answer [PMID 111].", _VALID_PMIDS)
        assert isinstance(entry["selected_sentences"], list)
        assert isinstance(entry["selected_sentences"][0], dict)
        assert "pmid" in entry["selected_sentences"][0]
        assert "sentence" in entry["selected_sentences"][0]


class TestBulkPipelineJsonRoundTrip:
    """Group 10b: bulk results survive JSON serialization."""

    def test_json_round_trip(self, tmp_path):
        entries = [
            _make_bulk_entry("1", "Q1?", "Answer one [PMID 111].", _VALID_PMIDS),
            _make_bulk_entry("2", "Q2?", "Answer two [PMID 222].", _VALID_PMIDS),
        ]
        fpath = tmp_path / "test_answers.json"
        with open(fpath, "w") as f:
            json.dump(entries, f, indent=2)
        with open(fpath) as f:
            loaded = json.load(f)
        assert len(loaded) == 2
        assert loaded[0]["query_id"] == "1"
        assert loaded[1]["parsed"]["word_count"] == 3  # "Answer two ."

    def test_json_round_trip_preserves_violations(self, tmp_path):
        raw = " ".join(["word"] * 260) + " [PMID 111]."
        entry = _make_bulk_entry("1", "Q?", raw, _VALID_PMIDS)
        fpath = tmp_path / "test.json"
        with open(fpath, "w") as f:
            json.dump([entry], f)
        with open(fpath) as f:
            loaded = json.load(f)
        assert loaded[0]["parsed"]["violations"]["over_word_limit"] is True


class TestConstraintStatsComputation:
    """Group 10c: constraint compliance statistics (Table 3 logic)."""

    def test_all_pass(self):
        entries = [
            _make_bulk_entry("1", "Q?", "Answer [PMID 111].", _VALID_PMIDS),
            _make_bulk_entry("2", "Q?", "Answer [PMID 222].", _VALID_PMIDS),
        ]
        n_pass = sum(check_constraints(e["parsed"]) for e in entries)
        assert n_pass == 2

    def test_one_fails_word_limit(self):
        ok = _make_bulk_entry("1", "Q?", "Short [PMID 111].", _VALID_PMIDS)
        bad = _make_bulk_entry("2", "Q?", " ".join(["w"] * 260) + " [PMID 111].", _VALID_PMIDS)
        entries = [ok, bad]
        n_pass = sum(check_constraints(e["parsed"]) for e in entries)
        assert n_pass == 1

    def test_one_fails_invalid_pmid(self):
        ok = _make_bulk_entry("1", "Q?", "OK [PMID 111].", _VALID_PMIDS)
        bad = _make_bulk_entry("2", "Q?", "Bad [PMID 999].", _VALID_PMIDS)
        entries = [ok, bad]
        n_pass = sum(check_constraints(e["parsed"]) for e in entries)
        assert n_pass == 1

    def test_word_count_stat(self):
        # Note: "[PMID X]." leaves trailing "." as a word token after stripping
        entries = [
            _make_bulk_entry("1", "Q?", "One two three[PMID 111].", _VALID_PMIDS),
            _make_bulk_entry("2", "Q?", "A B C D E[PMID 222].", _VALID_PMIDS),
        ]
        word_counts = [e["parsed"]["word_count"] for e in entries]
        assert word_counts == [3, 5]

    def test_pmids_per_sentence_stat(self):
        entries = [
            _make_bulk_entry("1", "Q?",
                "First [PMID 111]. Second [PMID 222, PMID 333].", _VALID_PMIDS),
        ]
        pmids_per_sent = [
            len(s["pmids"])
            for e in entries
            for s in e["parsed"]["sentences"]
        ]
        assert pmids_per_sent == [1, 2]


class TestGenerateAnswerRetry:
    """Group 10d: generate_answer retry on rate-limit errors."""

    def test_retry_on_429(self):
        mock_client = mock.MagicMock()
        # Fail twice with 429, succeed on third attempt
        mock_client.chat.completions.create.side_effect = [
            RuntimeError("IAedu API error: Rate limit reached (429)"),
            RuntimeError("IAedu API error: Rate limit reached (429)"),
            mock.MagicMock(choices=[mock.MagicMock(
                message=mock.MagicMock(content="OK [PMID 1].")
            )]),
        ]
        result = generate_answer("Q?", "ctx", mock_client, "m",
                                 max_retries=5, backoff_base=0.01)
        assert result == "OK [PMID 1]."
        assert mock_client.chat.completions.create.call_count == 3

    def test_non_429_error_propagates_immediately(self):
        mock_client = mock.MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("Connection refused")
        with pytest.raises(RuntimeError, match="Connection refused"):
            generate_answer("Q?", "ctx", mock_client, "m",
                            max_retries=3, backoff_base=0.01)
        assert mock_client.chat.completions.create.call_count == 1

    def test_exhausted_retries_raises(self):
        mock_client = mock.MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError(
            "Rate limit reached (429)")
        with pytest.raises(RuntimeError, match="Failed after 2 retries"):
            generate_answer("Q?", "ctx", mock_client, "m",
                            max_retries=2, backoff_base=0.01)
