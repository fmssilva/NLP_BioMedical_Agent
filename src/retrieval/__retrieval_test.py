"""
Tests for all retrieval strategies.
!!!! SET OF TESTES GENERATED WITH AI HELP - COVERS BASIC FUNCTIONALITY AND EDGE CASES FOR BOTH MODULES. 

Two categories:
  1. Unit tests  — no OpenSearch needed; mock the client and verify request bodies,
                   field name derivation, parameter validation, and response parsing.
  2. Integration — hit a live OpenSearch index with a small multi-field corpus,
                   confirm every field variant actually returns results.

Run:
    python -m src.retrieval.__retrieval_test

Unit tests cover:
  _extract_hits helpers
  _float_tag and _embedding_field derivation functions
  BM25       : default field ("contents"), custom k1/b field name, request body
  SparseRetriever: _build_body, match_phrase, invalid match_type raises ValueError
  BaseRetriever.field_exists classmethod (mocked) — also callable via KNNRetriever
  LMJM  : field derived from lam float, invalid lam raises ValueError
  LMDir : field derived from mu int, invalid mu raises ValueError
  KNN   : embed_field derived from alias, TypeError on bad encoder type,
          request body has correct field name and k==size
  MedCPTKNNRetriever: CLS pooling (not mean), max_length=64, targets embedding_medcpt field
  rrf_merge: disjoint, overlapping, custom k, score range
  RRFRetriever: sorted, truncated, no duplicates

Integration tests cover:
  Build a throwaway index with all variants, index 5 synthetic docs, run every
  retriever, assert results are non-empty, sorted, duplicate-free.
  Tear down the index at the end.
"""

import logging
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.embeddings.encoder import Encoder
from src.retrieval.base import BaseRetriever, SparseRetriever, _extract_hits
from src.indexing.index_builder import float_tag
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.knn import KNNRetriever, MedCPTKNNRetriever, _embedding_field
from src.retrieval.lm_dirichlet import LMDirichletRetriever
from src.retrieval.lm_jelinek_mercer import LMJMRetriever
from src.retrieval.rrf import RRFRetriever, rrf_merge
from src.evaluation.evaluator import build_query

logging.basicConfig(level=logging.WARNING)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_hit(doc_id: str, score: float) -> dict:
    return {"_id": doc_id, "_source": {"doc_id": doc_id}, "_score": score}

def _make_response(*hits: tuple[str, float]) -> dict:
    return {"hits": {"hits": [_make_hit(d, s) for d, s in hits]}}

def _make_client(*hits: tuple[str, float]):
    c = MagicMock()
    c.search.return_value = _make_response(*hits)
    return c


# ---------------------------------------------------------------------------
# 1. _extract_hits
# ---------------------------------------------------------------------------

class TestExtractHits(unittest.TestCase):

    def test_normal_hits(self):
        hits = [_make_hit("A", 1.5), _make_hit("B", 0.8)]
        self.assertEqual(_extract_hits(hits), [("A", 1.5), ("B", 0.8)])

    def test_missing_doc_id_skipped_with_warning(self):
        hits = [{"_id": "X", "_source": {}, "_score": 1.0}, _make_hit("B", 0.5)]
        with self.assertLogs("src.retrieval.base", level="WARNING"):
            result = _extract_hits(hits)
        self.assertEqual(result, [("B", 0.5)])

    def test_empty(self):
        self.assertEqual(_extract_hits([]), [])


# ---------------------------------------------------------------------------
# 2. Field/alias derivation helpers
# ---------------------------------------------------------------------------

class TestDerivationHelpers(unittest.TestCase):

    # float_tag (shared from index_builder -- used by bm25.py and lmjm.py)
    def test_float_tag_removes_dot(self):
        self.assertEqual(float_tag(0.7),  "07")
        self.assertEqual(float_tag(1.5),  "15")
        self.assertEqual(float_tag(1.2),  "12")
        self.assertEqual(float_tag(0.75), "075")
        self.assertEqual(float_tag(0.1),  "01")
        self.assertEqual(float_tag(0.4),  "04")

    # _embedding_field -- all aliases get embedding_{alias}, including msmarco
    def test_embedding_field_msmarco_is_explicit(self):
        self.assertEqual(_embedding_field("msmarco"), "embedding_msmarco")

    def test_embedding_field_other_aliases(self):
        self.assertEqual(_embedding_field("medcpt"),   "embedding_medcpt")
        self.assertEqual(_embedding_field("multi-qa"), "embedding_multi-qa")
        self.assertEqual(_embedding_field("pubmed"),   "embedding_pubmed")


# ---------------------------------------------------------------------------
# 3. BM25Retriever — unit tests
# ---------------------------------------------------------------------------

class TestBM25RetrieverUnit(unittest.TestCase):

    def setUp(self):
        self.client = _make_client(("P1", 9.0), ("P2", 5.0), ("P3", 1.0))

    def test_default_params_use_explicit_field(self):
        r = BM25Retriever(self.client, "idx")
        self.assertEqual(r.field, "contents_bm25_k12_b075")

    def test_custom_k1_b_derives_field_name(self):
        r = BM25Retriever(self.client, "idx", k1=1.5, b=1.0)
        self.assertEqual(r.field, "contents_bm25_k15_b10")

    def test_another_custom_pair(self):
        r = BM25Retriever(self.client, "idx", k1=2.0, b=0.5)
        self.assertEqual(r.field, "contents_bm25_k20_b05")

    def test_request_body_uses_derived_field(self):
        r = BM25Retriever(self.client, "idx", k1=1.5, b=1.0)
        r.search("sleep apnea", size=10)
        body = self.client.search.call_args.kwargs.get("body") or self.client.search.call_args.args[0]
        self.assertIn("match", body["query"])
        self.assertIn("contents_bm25_k15_b10", body["query"]["match"])
        self.assertEqual(body["query"]["match"]["contents_bm25_k15_b10"]["query"], "sleep apnea")
        self.assertEqual(body["size"], 10)
        self.assertEqual(body["_source"], ["doc_id"])

    def test_default_request_body_uses_explicit_field(self):
        r = BM25Retriever(self.client, "idx")
        r.search("q", size=5)
        body = self.client.search.call_args.kwargs.get("body") or self.client.search.call_args.args[0]
        self.assertIn("contents_bm25_k12_b075", body["query"]["match"])

    def test_returns_correct_results(self):
        r = BM25Retriever(self.client, "idx")
        results = r.search("q")
        self.assertEqual(results, [("P1", 9.0), ("P2", 5.0), ("P3", 1.0)])


# ---------------------------------------------------------------------------
# 4. SparseRetriever — _build_body and match_phrase
#    field_exists is on BaseRetriever, so it works via SparseRetriever AND KNNRetriever
# ---------------------------------------------------------------------------

class TestFieldRetrieverUnit(unittest.TestCase):

    def setUp(self):
        self.client = _make_client(("A", 2.0), ("B", 1.0))

    def test_build_body_match(self):
        r = SparseRetriever(self.client, "idx", "myfield")
        body = r._build_body("hello", 7)
        self.assertEqual(body["size"], 7)
        self.assertEqual(body["_source"], ["doc_id"])
        self.assertIn("match", body["query"])
        self.assertEqual(body["query"]["match"]["myfield"]["query"], "hello")

    def test_build_body_match_phrase(self):
        r = SparseRetriever(self.client, "idx", "myfield", match_type="match_phrase")
        body = r._build_body("hello world", 5)
        self.assertIn("match_phrase", body["query"])
        self.assertNotIn("match", body["query"])

    def test_invalid_match_type_raises_value_error(self):
        with self.assertRaises(ValueError):
            SparseRetriever(self.client, "idx", "f", match_type="multi_match")

    def test_search_uses_build_body(self):
        r = SparseRetriever(self.client, "idx", "myfield")
        r.search("q", size=3)
        body = self.client.search.call_args.kwargs.get("body") or self.client.search.call_args.args[0]
        self.assertEqual(body["size"], 3)
        self.assertIn("myfield", body["query"]["match"])

    # field_exists classmethod
    def test_field_exists_returns_true(self):
        client = MagicMock()
        client.indices.exists.return_value = True
        client.indices.get_mapping.return_value = {
            "myindex": {"mappings": {"properties": {"contents": {"type": "text"}}}}
        }
        self.assertTrue(SparseRetriever.field_exists(client, "myindex", "contents"))

    def test_field_exists_returns_false_missing_field(self):
        client = MagicMock()
        client.indices.exists.return_value = True
        client.indices.get_mapping.return_value = {
            "myindex": {"mappings": {"properties": {"contents": {"type": "text"}}}}
        }
        self.assertFalse(SparseRetriever.field_exists(client, "myindex", "no_such_field"))

    def test_field_exists_returns_false_no_index(self):
        client = MagicMock()
        client.indices.exists.return_value = False
        self.assertFalse(SparseRetriever.field_exists(client, "noindex", "contents"))

    def test_assert_field_exists_raises(self):
        client = MagicMock()
        client.indices.exists.return_value = True
        client.indices.get_mapping.return_value = {
            "idx": {"mappings": {"properties": {}}}
        }
        with self.assertRaises(ValueError) as ctx:
            SparseRetriever.assert_field_exists(client, "idx", "missing_field")
        self.assertIn("missing_field", str(ctx.exception))

    def test_field_exists_callable_from_knn_retriever(self):
        # field_exists lives on BaseRetriever, so KNNRetriever inherits it too
        client = MagicMock()
        client.indices.exists.return_value = True
        client.indices.get_mapping.return_value = {
            "myindex": {"mappings": {"properties": {"embedding": {"type": "knn_vector"}}}}
        }
        self.assertTrue(KNNRetriever.field_exists(client, "myindex", "embedding"))
        self.assertFalse(KNNRetriever.field_exists(client, "myindex", "embedding_medcpt"))


# ---------------------------------------------------------------------------
# 5. LMJMRetriever — free-form lambda
# ---------------------------------------------------------------------------

class TestLMJMRetrieverUnit(unittest.TestCase):

    def setUp(self):
        self.client = _make_client(("P1", 3.0), ("P2", 1.5))

    def test_lam_01_field(self):
        r = LMJMRetriever(self.client, "idx", lambd=0.1)
        self.assertEqual(r.field, "contents_lmjm_01")

    def test_lam_07_field(self):
        r = LMJMRetriever(self.client, "idx", lambd=0.7)
        self.assertEqual(r.field, "contents_lmjm_07")

    def test_lam_04_field(self):
        r = LMJMRetriever(self.client, "idx", lambd=0.4)
        self.assertEqual(r.field, "contents_lmjm_04")

    def test_default_lam_is_07(self):
        r = LMJMRetriever(self.client, "idx")
        self.assertEqual(r.field, "contents_lmjm_07")

    def test_invalid_lam_zero_raises(self):
        with self.assertRaises(ValueError):
            LMJMRetriever(self.client, "idx", lambd=0.0)

    def test_invalid_lam_one_raises(self):
        with self.assertRaises(ValueError):
            LMJMRetriever(self.client, "idx", lambd=1.0)

    def test_invalid_lam_negative_raises(self):
        with self.assertRaises(ValueError):
            LMJMRetriever(self.client, "idx", lambd=-0.1)

    def test_request_body_uses_correct_field(self):
        r = LMJMRetriever(self.client, "idx", lambd=0.4)
        r.search("q", size=10)
        body = self.client.search.call_args.kwargs.get("body") or self.client.search.call_args.args[0]
        self.assertIn("contents_lmjm_04", body["query"]["match"])

    def test_returns_results(self):
        r = LMJMRetriever(self.client, "idx", lambd=0.7)
        self.assertEqual(r.search("q"), [("P1", 3.0), ("P2", 1.5)])


# ---------------------------------------------------------------------------
# 6. LMDirichletRetriever — multi-mu
# ---------------------------------------------------------------------------

class TestLMDirichletRetrieverUnit(unittest.TestCase):

    def setUp(self):
        self.client = _make_client(("D1", 4.0), ("D2", 2.0))

    def test_default_mu_2000_field(self):
        r = LMDirichletRetriever(self.client, "idx")
        self.assertEqual(r.field, "contents_lmdir_2000")

    def test_mu_75_field(self):
        r = LMDirichletRetriever(self.client, "idx", mu=75)
        self.assertEqual(r.field, "contents_lmdir_75")

    def test_mu_500_field(self):
        r = LMDirichletRetriever(self.client, "idx", mu=500)
        self.assertEqual(r.field, "contents_lmdir_500")

    def test_invalid_mu_zero_raises(self):
        with self.assertRaises(ValueError):
            LMDirichletRetriever(self.client, "idx", mu=0)

    def test_invalid_mu_negative_raises(self):
        with self.assertRaises(ValueError):
            LMDirichletRetriever(self.client, "idx", mu=-100)

    def test_request_body_uses_correct_field(self):
        r = LMDirichletRetriever(self.client, "idx", mu=75)
        r.search("q", size=5)
        body = self.client.search.call_args.kwargs.get("body") or self.client.search.call_args.args[0]
        self.assertIn("contents_lmdir_75", body["query"]["match"])

    def test_returns_results(self):
        r = LMDirichletRetriever(self.client, "idx", mu=75)
        self.assertEqual(r.search("q"), [("D1", 4.0), ("D2", 2.0)])


# ---------------------------------------------------------------------------
# 7. KNNRetriever — multi-alias, field name, request body
# ---------------------------------------------------------------------------

class TestKNNRetrieverUnit(unittest.TestCase):

    def _make_enc(self, dim=768):
        enc = MagicMock(spec=Encoder)
        enc.encode_single.return_value = np.ones(dim, dtype=np.float32) / np.sqrt(dim)
        return enc

    def setUp(self):
        self.client = _make_client(("K1", 0.95), ("K2", 0.80))

    def test_default_alias_msmarco_uses_embedding_msmarco_field(self):
        r = KNNRetriever(self.client, "idx", encoder=self._make_enc())
        self.assertEqual(r.embed_field, "embedding_msmarco")
        self.assertEqual(r.encoder_alias, "msmarco")

    def test_medcpt_alias_uses_embedding_medcpt(self):
        r = KNNRetriever(self.client, "idx", encoder=self._make_enc(), encoder_alias="medcpt")
        self.assertEqual(r.embed_field, "embedding_medcpt")

    def test_multi_qa_alias(self):
        r = KNNRetriever(self.client, "idx", encoder=self._make_enc(), encoder_alias="multi-qa")
        self.assertEqual(r.embed_field, "embedding_multi-qa")

    def test_request_body_has_correct_field_and_k_equals_size(self):
        enc = self._make_enc(768)
        r = KNNRetriever(self.client, "idx", encoder=enc, encoder_alias="medcpt")
        r.search("sleep apnea", size=50)
        body = self.client.search.call_args.kwargs.get("body") or self.client.search.call_args.args[0]
        knn_block = body["query"]["knn"]["embedding_medcpt"]
        self.assertEqual(knn_block["k"], 50)               # k == size
        self.assertEqual(len(knn_block["vector"]), 768)
        self.assertEqual(body["size"], 50)

    def test_baseline_alias_body_uses_embedding_msmarco(self):
        enc = self._make_enc(768)
        r = KNNRetriever(self.client, "idx", encoder=enc)   # default alias = msmarco
        r.search("q", size=10)
        body = self.client.search.call_args.kwargs.get("body") or self.client.search.call_args.args[0]
        self.assertIn("embedding_msmarco", body["query"]["knn"])
        self.assertNotIn("embedding", body["query"]["knn"])

    def test_vector_is_list_of_floats(self):
        r = KNNRetriever(self.client, "idx", encoder=self._make_enc())
        r.search("q", size=5)
        body = self.client.search.call_args.kwargs.get("body") or self.client.search.call_args.args[0]
        vec = body["query"]["knn"]["embedding_msmarco"]["vector"]
        self.assertIsInstance(vec, list)
        self.assertIsInstance(vec[0], float)

    def test_bad_encoder_type_raises_type_error(self):
        with self.assertRaises(TypeError):
            KNNRetriever(self.client, "idx", encoder=["msmarco"])
        with self.assertRaises(TypeError):
            KNNRetriever(self.client, "idx", encoder="a_string")

    def test_none_encoder_accepted(self):
        # Passing None is allowed; __init__ will create a default Encoder internally.
        # We patch Encoder so it doesn't load a real model.
        with patch("src.retrieval.knn.Encoder") as MockEncoder:
            MockEncoder.return_value = MagicMock(spec=Encoder)
            r = KNNRetriever(self.client, "idx", encoder=None)
            MockEncoder.assert_called_once()

    def test_encode_single_called_once_per_search(self):
        enc = self._make_enc()
        r = KNNRetriever(self.client, "idx", encoder=enc)
        r.search("q", size=5)
        enc.encode_single.assert_called_once_with("q")

    def test_returns_results(self):
        r = KNNRetriever(self.client, "idx", encoder=self._make_enc())
        self.assertEqual(r.search("q", size=10), [("K1", 0.95), ("K2", 0.80)])


# ---------------------------------------------------------------------------
# 8. MedCPTKNNRetriever — unit tests (all mocked, no model download)
# ---------------------------------------------------------------------------

class TestMedCPTKNNRetrieverUnit(unittest.TestCase):
    """
    MedCPTKNNRetriever is a dedicated class for MedCPT asymmetric retrieval.
    It always uses ncbi/MedCPT-Query-Encoder at query time (CLS pooling,
    max_length=64) and always targets the embedding_medcpt index field.
    All tests mock out the transformer model so nothing is downloaded.
    """

    def _make_retriever_with_mocked_encoder(self):
        """
        Return a MedCPTKNNRetriever whose internal model is pre-populated with
        a mock that returns a fake last_hidden_state tensor.
        """
        import torch

        client = MagicMock()
        client.search.return_value = _make_response(("MC1", 0.97), ("MC2", 0.85))

        r = MedCPTKNNRetriever(client, "idx")

        # inject a fake tokenizer that returns a minimal encoded dict
        fake_tok = MagicMock()
        fake_tok.return_value = {
            "input_ids":      torch.zeros(1, 5, dtype=torch.long),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
        }
        r._tokenizer = fake_tok

        # inject a fake model whose __call__ returns an object with last_hidden_state
        hidden = torch.rand(1, 5, 768)   # (batch=1, seq_len=5, hidden=768)
        fake_out = MagicMock()
        fake_out.last_hidden_state = hidden

        fake_model = MagicMock()
        fake_model.return_value = fake_out
        r._model = fake_model
        r._device = "cpu"

        return r, client, hidden

    # -- lazy loading --

    def test_model_not_loaded_before_first_encode(self):
        r = MedCPTKNNRetriever(MagicMock(), "idx")
        self.assertIsNone(r._model)
        self.assertIsNone(r._tokenizer)

    # -- CLS pooling + L2 normalisation --

    def test_encode_query_uses_cls_token(self):
        """encode_query must return the CLS token (position 0), L2-normalised."""
        import torch
        import torch.nn.functional as F

        r, _client, hidden = self._make_retriever_with_mocked_encoder()
        vec = r.encode_query("sleep apnea")

        # expected: F.normalize(hidden[:, 0, :], p=2, dim=1)[0]
        expected = F.normalize(hidden[:, 0, :], p=2, dim=1)[0].numpy()
        np.testing.assert_allclose(vec, expected, atol=1e-6,
            err_msg="encode_query did not return the CLS (position-0) token")

    def test_encode_query_not_mean_pooled(self):
        """CLS result must differ from mean-pooled result for non-uniform hidden states."""
        import torch
        import torch.nn.functional as F

        r, _client, hidden = self._make_retriever_with_mocked_encoder()
        vec = r.encode_query("q")

        # compute what mean-pooling WOULD give — must differ
        mask = torch.ones(1, 5).unsqueeze(-1).expand(hidden.size()).float()
        mean_pooled = F.normalize(
            torch.sum(hidden * mask, 1) / torch.clamp(mask.sum(1), min=1e-9),
            p=2, dim=1
        )[0].numpy()

        self.assertFalse(np.allclose(vec, mean_pooled, atol=1e-4),
            "encode_query returned mean-pooled result — should be CLS")

    def test_encode_query_output_shape(self):
        r, _client, _hidden = self._make_retriever_with_mocked_encoder()
        vec = r.encode_query("test query")
        self.assertEqual(vec.shape, (768,))

    def test_encode_query_l2_normalised(self):
        r, _client, _hidden = self._make_retriever_with_mocked_encoder()
        vec = r.encode_query("test query")
        norm = float(np.linalg.norm(vec))
        self.assertAlmostEqual(norm, 1.0, places=5,
            msg=f"encode_query output not L2-normalised (norm={norm:.6f})")

    # -- tokenizer called with max_length=64 (official MedCPT spec) --

    def test_tokenizer_called_with_max_length_64(self):
        r, _client, _hidden = self._make_retriever_with_mocked_encoder()
        r.encode_query("sleep apnea")
        call_kwargs = r._tokenizer.call_args[1]
        self.assertEqual(call_kwargs.get("max_length"), 64,
            "MedCPT-Query-Encoder must tokenize queries with max_length=64")

    def test_tokenizer_called_with_truncation(self):
        r, _client, _hidden = self._make_retriever_with_mocked_encoder()
        r.encode_query("q")
        call_kwargs = r._tokenizer.call_args[1]
        self.assertTrue(call_kwargs.get("truncation"),
            "truncation must be True when encoding queries")

    # -- search targets embedding_medcpt field --

    def test_search_targets_embedding_medcpt_field(self):
        r, client, _hidden = self._make_retriever_with_mocked_encoder()
        r.search("sleep apnea", size=10)
        body = client.search.call_args.kwargs.get("body") or client.search.call_args.args[0]
        self.assertIn("embedding_medcpt", body["query"]["knn"],
            "MedCPTKNNRetriever.search must query the 'embedding_medcpt' field")

    def test_search_k_equals_size(self):
        r, client, _hidden = self._make_retriever_with_mocked_encoder()
        r.search("q", size=50)
        body = client.search.call_args.kwargs.get("body") or client.search.call_args.args[0]
        self.assertEqual(body["query"]["knn"]["embedding_medcpt"]["k"], 50)
        self.assertEqual(body["size"], 50)

    def test_search_returns_results(self):
        r, _client, _hidden = self._make_retriever_with_mocked_encoder()
        results = r.search("q", size=10)
        self.assertEqual(results, [("MC1", 0.97), ("MC2", 0.85)])

    def test_search_vector_is_list_of_floats(self):
        r, client, _hidden = self._make_retriever_with_mocked_encoder()
        r.search("q", size=5)
        body = client.search.call_args.kwargs.get("body") or client.search.call_args.args[0]
        vec = body["query"]["knn"]["embedding_medcpt"]["vector"]
        self.assertIsInstance(vec, list)
        self.assertIsInstance(vec[0], float)


# ---------------------------------------------------------------------------
# 9. rrf_merge — pure function
# ---------------------------------------------------------------------------

class TestRRFMerge(unittest.TestCase):

    def test_disjoint_lists_all_docs_present(self):
        a = [("A", 1.0), ("B", 0.5)]
        b = [("C", 1.0), ("D", 0.5)]
        merged = rrf_merge(a, b, k=60)
        self.assertEqual(sorted(p for p, _ in merged), ["A", "B", "C", "D"])

    def test_disjoint_scores_descending(self):
        a = [("A", 1.0), ("B", 0.5)]
        b = [("C", 1.0), ("D", 0.5)]
        merged = rrf_merge(a, b, k=60)
        scores = [s for _, s in merged]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_overlapping_doc_ranked_higher(self):
        # B is #2 in A and #1 in B => highest combined score
        a = [("A", 1.0), ("B", 0.9), ("C", 0.8)]
        b = [("B", 1.0), ("D", 0.9), ("A", 0.8)]
        merged = rrf_merge(a, b, k=60)
        self.assertEqual(merged[0][0], "B")

    def test_custom_k(self):
        a = [("A", 1.0)]
        b = [("A", 1.0)]
        merged = rrf_merge(a, b, k=10)
        # rank 1 in both → 2 / (10+1)
        self.assertAlmostEqual(merged[0][1], 2.0 / 11, places=8)

    def test_scores_are_positive(self):
        a = [(str(i), float(10-i)) for i in range(10)]
        b = [(str(9-i), float(10-i)) for i in range(10)]
        for _, score in rrf_merge(a, b, k=60):
            self.assertGreater(score, 0.0)

    def test_no_duplicates_in_output(self):
        a = [("A", 1.0), ("B", 0.5)]
        b = [("A", 0.9), ("C", 0.4)]
        merged = rrf_merge(a, b, k=60)
        pmids = [p for p, _ in merged]
        self.assertEqual(len(pmids), len(set(pmids)))


# ---------------------------------------------------------------------------
# 10. RRFRetriever — any two mocked retrievers
# ---------------------------------------------------------------------------

class TestRRFRetrieverUnit(unittest.TestCase):

    def _mock_retriever(self, results):
        """Return a MagicMock that satisfies isinstance(x, BaseRetriever)."""
        m = MagicMock(spec=BaseRetriever)
        m.search.return_value = results
        return m

    # -- basic behaviour --

    def test_sorted_descending(self):
        rrf = RRFRetriever(
            self._mock_retriever([("A", 9.0), ("B", 5.0), ("C", 1.0)]),
            self._mock_retriever([("B", 0.9), ("D", 0.7), ("A", 0.5)]),
        )
        scores = [s for _, s in rrf.search("q", size=10)]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_truncated_to_size(self):
        rrf = RRFRetriever(
            self._mock_retriever([(f"A{i}", float(10-i)) for i in range(10)]),
            self._mock_retriever([(f"B{i}", float(10-i)) for i in range(10)]),
        )
        self.assertEqual(len(rrf.search("q", size=5)), 5)

    def test_no_duplicates(self):
        rrf = RRFRetriever(
            self._mock_retriever([("A", 9.0), ("B", 5.0)]),
            self._mock_retriever([("A", 0.9), ("C", 0.7)]),
        )
        pmids = [p for p, _ in rrf.search("q", size=10)]
        self.assertEqual(len(pmids), len(set(pmids)))

    def test_custom_rrf_k_stored(self):
        rrf = RRFRetriever(
            self._mock_retriever([]),
            self._mock_retriever([]),
            rrf_k=30,
        )
        self.assertEqual(rrf.rrf_k, 30)

    def test_bad_retriever_a_raises(self):
        with self.assertRaises(TypeError):
            RRFRetriever("not_a_retriever", self._mock_retriever([]))

    def test_bad_retriever_b_raises(self):
        with self.assertRaises(TypeError):
            RRFRetriever(self._mock_retriever([]), 42)

    # -- pair combination: overlapping docs scored correctly --

    def test_bm25_lmjm_pair(self):
        a = self._mock_retriever([("P1", 9.0), ("P2", 5.0)])
        b = self._mock_retriever([("P2", 3.0), ("P3", 1.0)])
        rrf = RRFRetriever(a, b)
        results = rrf.search("q", size=10)
        pmids = [p for p, _ in results]
        self.assertNotIn(len(pmids), [0])          # non-empty
        self.assertEqual(len(pmids), len(set(pmids)))  # no dupes

    def test_lmjm_lmdir_pair(self):
        a = self._mock_retriever([("X", 8.0), ("Y", 4.0)])
        b = self._mock_retriever([("Y", 7.0), ("Z", 2.0)])
        rrf = RRFRetriever(a, b)
        results = rrf.search("q", size=10)
        # Y is in both lists so it should outscore X and Z
        scores = {p: s for p, s in results}
        self.assertGreater(scores["Y"], scores["X"])
        self.assertGreater(scores["Y"], scores["Z"])

    def test_both_retrievers_called_with_query_and_size(self):
        a = self._mock_retriever([("A", 1.0)])
        b = self._mock_retriever([("B", 1.0)])
        rrf = RRFRetriever(a, b)
        rrf.search("sleep apnea", size=7)
        a.search.assert_called_once_with("sleep apnea", size=7)
        b.search.assert_called_once_with("sleep apnea", size=7)


# ---------------------------------------------------------------------------
# 11. build_query — evaluator helper
# ---------------------------------------------------------------------------

class TestBuildQuery(unittest.TestCase):

    def setUp(self):
        self.topic = {
            "id": 116,
            "topic": "obstructive sleep apnea",
            "question": "What is the best treatment for OSA?",
            "narrative": "Looking for RCTs.",
        }

    def test_topic_field(self):
        self.assertEqual(build_query(self.topic, "topic"), "obstructive sleep apnea")

    def test_question_field(self):
        self.assertEqual(build_query(self.topic, "question"), self.topic["question"])

    def test_concatenated_contains_all_parts(self):
        q = build_query(self.topic, "concatenated")
        for part in [self.topic["topic"], self.topic["question"], self.topic["narrative"]]:
            self.assertIn(part, q)

    def test_invalid_field_raises_value_error(self):
        with self.assertRaises(ValueError):
            build_query(self.topic, "abstract")

    def test_concatenated_longer_than_topic(self):
        self.assertGreater(
            len(build_query(self.topic, "concatenated")),
            len(build_query(self.topic, "topic")),
        )


# ---------------------------------------------------------------------------
# 11. Integration tests — live OpenSearch, existing usernlp03 index
# ---------------------------------------------------------------------------

def run_integration_tests():
    """
    Run against the live usernlp03 index that is already populated.

    Fields confirmed to exist in the index (from get_live_field_types()):
      BM25 explicit : "contents_bm25_k12_b075"  (k1=1.2, b=0.75)
      BM25 custom   : "contents_bm25_k15_b10"   (k1=1.5, b=1.0)
      LM-JM         : "contents_lmjm_01", "contents_lmjm_07"
      LM-Dir        : "contents_lmdir_75", "contents_lmdir_500"
      KNN msmarco   : "embedding_msmarco"
      KNN medcpt    : "embedding_medcpt"

    We do NOT create or delete any index (no admin rights on shared cluster).
    """
    from dotenv import load_dotenv
    from src.indexing.opensearch_client import get_client

    load_dotenv()
    client = get_client()
    index_name = os.getenv("OPENSEARCH_INDEX", "")
    assert index_name, "OPENSEARCH_INDEX env var not set"

    QUERY = "obstructive sleep apnea treatment"
    SIZE  = 100
    PASS  = "[ok]"
    FAIL  = "[FAIL]"
    passed = failed = 0

    def check(name, cond, detail=""):
        nonlocal passed, failed
        if cond:
            print(f"  {PASS} {name}")
            passed += 1
        else:
            print(f"  {FAIL} {name}  {detail}")
            failed += 1

    def run_check(label, retriever):
        results = retriever.search(QUERY, size=SIZE)
        scores  = [s for _, s in results]
        pmids   = [p for p, _ in results]
        check(f"{label}: non-empty",  len(results) > 0)
        check(f"{label}: sorted",     scores == sorted(scores, reverse=True))
        check(f"{label}: no dupes",   len(pmids) == len(set(pmids)))
        if results:
            print(f"    field={retriever.field if hasattr(retriever, 'field') else retriever.embed_field}  "
                  f"top-1={pmids[0]}  score={scores[0]:.4f}  count={len(results)}")
        return results

    print("=" * 60)
    print(f"Integration tests — index: {index_name}")
    print(f"  query : '{QUERY}'")
    print("=" * 60)

    # -- Connection + field_exists checks --
    print("\n-- Connectivity & field_exists --")
    count = client.count(index=index_name)["count"]
    check("Index reachable, docs > 0", count > 0, f"count={count}")

    for field in ["contents_bm25_k12_b075", "contents_bm25_k15_b10",
                  "contents_lmjm_01", "contents_lmjm_07",
                  "contents_lmdir_75", "contents_lmdir_500",
                  "embedding_msmarco", "embedding_medcpt"]:
        check(f"field_exists('{field}')",
              SparseRetriever.field_exists(client, index_name, field))

    check("field_exists('no_such_field') is False",
          not SparseRetriever.field_exists(client, index_name, "no_such_field"))

    # -- BM25 variants --
    print("\n-- BM25 (k1=1.2, b=0.75) → 'contents_bm25_k12_b075' --")
    run_check("BM25-default", BM25Retriever(client, index_name))

    print("\n-- BM25 (k1=1.5, b=1.0) → 'contents_bm25_k15_b10' --")
    run_check("BM25-k15-b10", BM25Retriever(client, index_name, k1=1.5, b=1.0))

    # -- LM-JM variants --
    print("\n-- LM-JM (lam=0.1) → 'contents_lmjm_01' --")
    run_check("LMJM-01", LMJMRetriever(client, index_name, lambd=0.1))

    print("\n-- LM-JM (lam=0.7) → 'contents_lmjm_07' --")
    run_check("LMJM-07", LMJMRetriever(client, index_name, lambd=0.7))

    # -- LM-Dir variants --
    print("\n-- LM-Dir (mu=75) → 'contents_lmdir_75' --")
    run_check("LMDir-75",  LMDirichletRetriever(client, index_name, mu=75))

    print("\n-- LM-Dir (mu=500) → 'contents_lmdir_500' --")
    run_check("LMDir-500", LMDirichletRetriever(client, index_name, mu=500))

    # -- KNN: msmarco → "embedding_msmarco" --
    print("\n-- KNN msmarco → 'embedding_msmarco' (~10s to load model) --")
    enc_msmarco = Encoder("sentence-transformers/msmarco-distilbert-base-v2")
    knn_ms = KNNRetriever(client, index_name, encoder=enc_msmarco, encoder_alias="msmarco")
    check("KNN msmarco embed_field == 'embedding_msmarco'", knn_ms.embed_field == "embedding_msmarco")
    r_knn_ms = knn_ms.search(QUERY, size=SIZE)
    scores_ms = [s for _, s in r_knn_ms]
    check("KNN-msmarco: non-empty",  len(r_knn_ms) > 0)
    check("KNN-msmarco: sorted",     scores_ms == sorted(scores_ms, reverse=True))
    check("KNN-msmarco: no dupes",   len(set(p for p,_ in r_knn_ms)) == len(r_knn_ms))
    check("KNN-msmarco: scores > 0", all(s > 0 for s in scores_ms))
    if r_knn_ms:
        print(f"    embed_field=embedding_msmarco  top-1={r_knn_ms[0][0]}  score={r_knn_ms[0][1]:.4f}  count={len(r_knn_ms)}")

    # -- KNN: medcpt → "embedding_medcpt" --
    print("\n-- KNN medcpt → 'embedding_medcpt' --")
    enc_medcpt = Encoder("ncbi/MedCPT-Query-Encoder")
    knn_mc = KNNRetriever(client, index_name, encoder=enc_medcpt, encoder_alias="medcpt")
    check("KNN medcpt embed_field == 'embedding_medcpt'", knn_mc.embed_field == "embedding_medcpt")
    r_knn_mc = knn_mc.search(QUERY, size=SIZE)
    scores_mc = [s for _, s in r_knn_mc]
    check("KNN-medcpt: non-empty",  len(r_knn_mc) > 0)
    check("KNN-medcpt: sorted",     scores_mc == sorted(scores_mc, reverse=True))
    check("KNN-medcpt: no dupes",   len(set(p for p,_ in r_knn_mc)) == len(r_knn_mc))
    check("KNN-medcpt: scores > 0", all(s > 0 for s in scores_mc))
    if r_knn_mc:
        print(f"    embed_field=embedding_medcpt  top-1={r_knn_mc[0][0]}  score={r_knn_mc[0][1]:.4f}  count={len(r_knn_mc)}")

    # -- RRF pairs against the live index --
    rrf_pairs = [
        ("BM25 + KNN-msmarco",
         BM25Retriever(client, index_name),
         KNNRetriever(client, index_name, encoder=enc_msmarco)),
        ("BM25 + LMJM-07",
         BM25Retriever(client, index_name),
         LMJMRetriever(client, index_name, lambd=0.7)),
        ("BM25 + LMDir-75",
         BM25Retriever(client, index_name),
         LMDirichletRetriever(client, index_name, mu=75)),
        ("LMJM-07 + LMDir-75",
         LMJMRetriever(client, index_name, lambd=0.7),
         LMDirichletRetriever(client, index_name, mu=75)),
    ]
    for label, a, b in rrf_pairs:
        print(f"\n-- RRF ({label}) --")
        rrf = RRFRetriever(a, b, rrf_k=60)
        r_rrf = rrf.search(QUERY, size=SIZE)
        scores_rrf = [s for _, s in r_rrf]
        check(f"RRF {label}: non-empty",      len(r_rrf) > 0)
        check(f"RRF {label}: sorted",         scores_rrf == sorted(scores_rrf, reverse=True))
        check(f"RRF {label}: no dupes",       len(set(p for p,_ in r_rrf)) == len(r_rrf))
        check(f"RRF {label}: scores in (0,1)", all(0 < s < 1 for s in scores_rrf))
        if r_rrf:
            print(f"    top-1={r_rrf[0][0]}  score={r_rrf[0][1]:.5f}  count={len(r_rrf)}")

    # -- assert_field_exists raises on bad field --
    print("\n-- assert_field_exists guard --")
    try:
        SparseRetriever.assert_field_exists(client, index_name, "contents_lmjm_99")
        check("assert_field_exists raises on missing field", False, "no exception raised")
    except ValueError:
        check("assert_field_exists raises on missing field", True)

    print(f"\n{'='*60}")
    print(f"Integration: {passed} passed / {failed} failed")
    print("=" * 60)
    if failed:
        print("[FAIL] Some integration tests failed.")
    else:
        print("[ok] All integration tests passed.")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("retrieval unit tests")
    print("=" * 60)

    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [
        TestExtractHits,
        TestDerivationHelpers,
        TestBM25RetrieverUnit,
        TestFieldRetrieverUnit,
        TestLMJMRetrieverUnit,
        TestLMDirichletRetrieverUnit,
        TestKNNRetrieverUnit,
        TestMedCPTKNNRetrieverUnit,
        TestRRFMerge,
        TestRRFRetrieverUnit,
        TestBuildQuery,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if not result.wasSuccessful():
        print("\n[FAIL] Unit tests failed. Fix before running integration tests.")
        sys.exit(1)

    print("\n[ok] All unit tests passed.\n")
    run_integration_tests()
