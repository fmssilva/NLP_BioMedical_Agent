"""
Local tests for qrels_builder.py.
Run with: python -m src.data.__qrels_builder_test
"""

import json
from pathlib import Path

from src.data.loader import load_corpus
from src.data.qrels_builder import (
    _DEFAULT_GRADED_SCORE,
    build_qrels,
    build_qrels_graded,
    rescale_qrels_graded,
)


ROOT = Path(__file__).resolve().parents[2]
SUBMISSIONS = ROOT / "data" / "biogen_2024_submissions.json"
CORPUS_PATH = ROOT / "data" / "filtered_pubmed_abstracts.txt"
BINARY_JSON = ROOT / "results" / "qrels" / "qrels.json"
GRADED_JSON = ROOT / "results" / "qrels" / "qrels_graded.json"

# ── Helpers ────────────────────────────────────────────────────────────────

def _load_corpus_pmids() -> set:
    corpus = load_corpus(CORPUS_PATH)
    return {doc["id"] for doc in corpus}


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Tests on the saved JSON files (fast: no re-build) ─────────────────────

def test_saved_structure():
    # qrels.json must be dict[str, dict[str, int]]
    qrels = _load_json(BINARY_JSON)
    assert isinstance(qrels, dict), "qrels.json root must be a dict"
    for qid, docs in qrels.items():
        assert isinstance(qid, str), f"query_id must be str, got {type(qid)}"
        assert isinstance(docs, dict), f"topic {qid} value must be dict"
        for pmid, score in docs.items():
            assert isinstance(pmid, str), f"doc_id must be str in topic {qid}"
            assert score == 1, f"Binary qrels must have score=1, got {score} for topic {qid} PMID {pmid}"
    print(f"  [OK] binary structure: {len(qrels)} topics, all scores == 1")


def test_graded_structure():
    # qrels_graded.json must be dict[str, dict[str, int]] with scores in {1,2}
    qrels = _load_json(GRADED_JSON)
    assert isinstance(qrels, dict), "qrels_graded.json root must be a dict"
    for qid, docs in qrels.items():
        assert isinstance(qid, str), f"query_id must be str, got {type(qid)}"
        assert isinstance(docs, dict), f"topic {qid} value must be dict"
        for pmid, score in docs.items():
            assert isinstance(pmid, str), f"doc_id must be str in topic {qid}"
            assert score in (1, 2), (
                f"Graded score must be 1 or 2, got {score} for topic {qid} PMID {pmid}"
            )
    print(f"  [OK] graded structure: {len(qrels)} topics, all scores in {{1,2}}")


def test_graded_superset_of_binary():
    # Every (topic, PMID) in binary must be in graded with score==2.
    # Graded may have extra entries (neutral, score=1).
    binary = _load_json(BINARY_JSON)
    graded = _load_json(GRADED_JSON)

    assert len(graded) >= len(binary), (
        f"Graded must have >= topics as binary: {len(graded)} vs {len(binary)}"
    )

    for qid, pmid_dict in binary.items():
        assert qid in graded, f"Topic {qid} in binary but not in graded"
        for pmid in pmid_dict:
            assert pmid in graded[qid], (
                f"Topic {qid} PMID {pmid}: in binary but missing from graded"
            )
            assert graded[qid][pmid] == 2, (
                f"Topic {qid} PMID {pmid}: in binary so must be score=2 in graded, got {graded[qid][pmid]}"
            )
    print(f"  [OK] graded is superset of binary: all binary PMIDs -> graded score=2")


def test_binary_derivable_from_graded():
    # The binary view can be derived from graded by filtering score >= 2.
    # This verifies the "single source of truth" approach.
    binary = _load_json(BINARY_JSON)
    graded = _load_json(GRADED_JSON)

    derived_binary = {
        qid: {pmid: 1 for pmid, s in docs.items() if s >= 2}
        for qid, docs in graded.items()
    }
    # Remove empty topics (shouldn't happen, but be safe)
    derived_binary = {qid: docs for qid, docs in derived_binary.items() if docs}

    assert set(derived_binary.keys()) == set(binary.keys()), (
        f"Topic mismatch: derived={set(derived_binary)-set(binary)} extra, "
        f"missing={set(binary)-set(derived_binary)}"
    )
    for qid in binary:
        assert set(derived_binary[qid].keys()) == set(binary[qid].keys()), (
            f"Topic {qid}: PMID set differs between derived and binary qrels"
        )
    print(f"  [OK] binary is perfectly derivable from graded with score>=2 threshold")


def test_corpus_membership():
    # Every PMID in both qrel files must exist in the corpus.
    corpus_pmids = _load_corpus_pmids()
    binary = _load_json(BINARY_JSON)
    graded = _load_json(GRADED_JSON)

    for label, qrels in [("binary", binary), ("graded", graded)]:
        for qid, docs in qrels.items():
            for pmid in docs:
                assert pmid in corpus_pmids, (
                    f"[{label}] Topic {qid} PMID {pmid} NOT in corpus (should have been filtered)"
                )
    print(f"  [OK] all qrel PMIDs are in corpus (4194 docs)")


def test_score_distribution():
    # Sanity check: expected approximate score counts based on known data.
    graded = _load_json(GRADED_JSON)
    score_counts = {1: 0, 2: 0}
    for docs in graded.values():
        for s in docs.values():
            score_counts[s] = score_counts.get(s, 0) + 1

    total = sum(score_counts.values())
    print(f"  Graded score distribution: score=2: {score_counts[2]}, score=1: {score_counts[1]}, total: {total}")
    # from the data: ~2999 supporting (score=2), ~218 neutral (score=1), total ~3217
    assert score_counts[2] > 2000, f"Too few score=2 entries: {score_counts[2]}"
    assert score_counts[1] > 0,   f"No neutral (score=1) entries found — unexpected"
    assert total == score_counts[1] + score_counts[2], "Score counts don't add up"
    print(f"  [OK] score distribution in expected range")


def test_no_empty_topics():
    # Every topic must have at least 1 relevant doc (otherwise it was filtered out at build time).
    for label, path in [("binary", BINARY_JSON), ("graded", GRADED_JSON)]:
        qrels = _load_json(path)
        for qid, docs in qrels.items():
            assert docs, f"[{label}] Topic {qid} has empty doc dict"
    print(f"  [OK] no empty topics in either qrels file")


def test_all_65_topics_present():
    # All 65 topics should have at least some relevant docs.
    binary = _load_json(BINARY_JSON)
    graded = _load_json(GRADED_JSON)
    assert len(binary) == 65, f"Expected 65 topics in binary qrels, got {len(binary)}"
    assert len(graded) == 65, f"Expected 65 topics in graded qrels, got {len(graded)}"
    print(f"  [OK] both qrel files have 65 topics (full dataset)")


# ── Tests by re-building from scratch (verifies qrels_builder logic) ───────

def test_rebuild_binary_matches_saved(corpus_pmids: set):
    # Re-run build_qrels from raw submissions, check it matches the saved file.
    saved = _load_json(BINARY_JSON)
    rebuilt = build_qrels(SUBMISSIONS, corpus_pmids=corpus_pmids)

    assert set(rebuilt.keys()) == set(saved.keys()), (
        f"Topic mismatch after rebuild: rebuilt has {len(rebuilt)}, saved has {len(saved)}"
    )
    for qid in saved:
        assert set(rebuilt[qid].keys()) == set(saved[qid].keys()), (
            f"Topic {qid}: PMID set differs after rebuild"
        )
    print(f"  [OK] rebuild of binary qrels is deterministic and matches saved file")


def test_rebuild_graded_matches_saved(corpus_pmids: set):
    # Re-run build_qrels_graded from raw submissions, check it matches the saved file.
    saved = _load_json(GRADED_JSON)
    rebuilt = build_qrels_graded(SUBMISSIONS, corpus_pmids=corpus_pmids)

    assert set(rebuilt.keys()) == set(saved.keys()), (
        f"Topic mismatch after graded rebuild"
    )
    for qid in saved:
        assert set(rebuilt[qid].keys()) == set(saved[qid].keys()), (
            f"Topic {qid}: PMID set differs after graded rebuild"
        )
        for pmid, score in saved[qid].items():
            assert rebuilt[qid][pmid] == score, (
                f"Topic {qid} PMID {pmid}: score {rebuilt[qid][pmid]} != saved {score}"
            )
    print(f"  [OK] rebuild of graded qrels is deterministic and matches saved file")


def test_graded_score_mapping():
    # Verify the _DEFAULT_GRADED_SCORE constant has exactly the expected entries.
    assert _DEFAULT_GRADED_SCORE["supporting"]       == 2
    assert _DEFAULT_GRADED_SCORE["neutral"]          == 1
    assert _DEFAULT_GRADED_SCORE["not relevant"]     == 0
    assert _DEFAULT_GRADED_SCORE["contradicting"]    == 0
    assert _DEFAULT_GRADED_SCORE["invalid citation"] == 0
    print(f"  [OK] _DEFAULT_GRADED_SCORE mapping is correct")


# ── Small synthetic test (no file I/O) ────────────────────────────────────

def test_synthetic_build():
    """
    Build qrels from a tiny hand-crafted submission and verify output exactly.
    Doesn't touch the real data files — pure unit test of the builder logic.
    """
    import tempfile, json as _json
    from src.data.qrels_builder import build_qrels, build_qrels_graded

    fake_submissions = [
        {
            "question_id": "999",
            "machine_generated_answers": {
                "sysA": {
                    "answer": "...",
                    "answer_sentences": [
                        {
                            "answer_sentence_id": "1",
                            "answer_sentence": "...",
                            "answer_sentence_relevance": "required",
                            "citation_assessment": [
                                {"cited_pmid": "AAA", "evidence_support": "x", "evidence_relation": "supporting"},
                                {"cited_pmid": "BBB", "evidence_support": "x", "evidence_relation": "neutral"},
                                {"cited_pmid": "CCC", "evidence_support": "x", "evidence_relation": "contradicting"},
                            ],
                        }
                    ],
                },
                "sysB": {
                    "answer": "...",
                    "answer_sentences": [
                        {
                            "answer_sentence_id": "1",
                            "answer_sentence": "...",
                            "answer_sentence_relevance": "required",
                            "citation_assessment": [
                                # AAA appears again as neutral in sysB — graded should keep max (=2)
                                {"cited_pmid": "AAA", "evidence_support": "x", "evidence_relation": "neutral"},
                                # DDD appears only in sysB as supporting
                                {"cited_pmid": "DDD", "evidence_support": "x", "evidence_relation": "supporting"},
                            ],
                        }
                    ],
                },
            },
        }
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tmp:
        _json.dump(fake_submissions, tmp)
        tmp_path = Path(tmp.name)

    # Binary: only supporting docs -> AAA and DDD
    binary = build_qrels(tmp_path)
    assert "999" in binary
    assert set(binary["999"].keys()) == {"AAA", "DDD"}, f"Binary wrong: {binary['999']}"
    assert all(s == 1 for s in binary["999"].values())

    # Graded: AAA=2 (supporting wins over neutral), BBB=1 (neutral), DDD=2 (supporting)
    graded = build_qrels_graded(tmp_path)
    assert "999" in graded
    assert graded["999"]["AAA"] == 2, f"AAA should be 2, got {graded['999']['AAA']}"
    assert graded["999"]["BBB"] == 1, f"BBB should be 1 (neutral), got {graded['999']['BBB']}"
    assert graded["999"]["DDD"] == 2, f"DDD should be 2, got {graded['999']['DDD']}"
    assert "CCC" not in graded["999"], "CCC is contradicting — should not appear in graded (score=0 filtered)"

    tmp_path.unlink()
    print(f"  [OK] synthetic build test: binary and graded correctly derived from fake submissions")


# ── rescale_qrels_graded tests (synthetic, no file I/O) ───────────────────

def test_rescale_0_2_to_0_5():
    # 0-2 scale -> 0-5: factor = 5/2 = 2.5
    # score 1 -> round(1*2.5) = round(2.5) = 2 (Python banker's rounding: round half to even)
    # score 2 -> round(2*2.5) = 5
    qrels = {"q1": {"A": 2, "B": 1}}
    out = rescale_qrels_graded(qrels, max_score_new=5)
    assert out["q1"]["A"] == 5, f"score 2->5 failed: got {out['q1']['A']}"
    assert out["q1"]["B"] == 2, f"score 1->2 failed: got {out['q1']['B']}"
    print(f"  [OK] rescale 0-2->0-5: score 2 -> 5, score 1 -> 2")

def test_rescale_0_2_to_0_7():
    # 0-2 scale -> 0-7: factor = 7/2 = 3.5
    # score 1 -> round(1*3.5) = round(3.5) = 4 (Python banker's rounding: round half to even)
    # score 2 -> round(2*3.5) = 7
    qrels = {"q1": {"A": 2, "B": 1}}
    out = rescale_qrels_graded(qrels, max_score_new=7)
    assert out["q1"]["A"] == 7, f"score 2->7 failed: got {out['q1']['A']}"
    assert out["q1"]["B"] == 4, f"score 1->4 failed: got {out['q1']['B']}"
    print(f"  [OK] rescale 0-2->0-7: score 2 -> 7, score 1 -> 4")

def test_rescale_identity():
    # rescale to same max -> scores unchanged
    qrels = {"q1": {"A": 2, "B": 1}}
    out = rescale_qrels_graded(qrels, max_score_new=2, max_score_orig=2)
    assert out["q1"]["A"] == 2
    assert out["q1"]["B"] == 1
    print(f"  [OK] rescale identity (0-2->0-2): scores unchanged")

def test_rescale_preserves_topic_structure():
    # all topics/PMIDs with score>=1 must be preserved
    qrels = {"q1": {"A": 2, "B": 1}, "q2": {"C": 2}}
    out = rescale_qrels_graded(qrels, max_score_new=5)
    assert set(out.keys()) == {"q1", "q2"}
    assert "A" in out["q1"] and "B" in out["q1"]
    assert "C" in out["q2"]
    print(f"  [OK] rescale preserves all topics and PMIDs with score>=1")

def test_rescale_filters_zeroed_scores():
    # if a score would rescale to 0, it should be filtered out
    # With max_new=1, max_orig=2: score 1 -> round(0.5)=0 (banker's rounding), score 2->1
    qrels = {"q1": {"A": 2, "B": 1}}
    out = rescale_qrels_graded(qrels, max_score_new=1, max_score_orig=2)
    # A (score=2) -> 1, kept; B (score=1) -> round(0.5)=0, dropped
    assert "A" in out["q1"], f"A should be kept with score 1"
    assert out["q1"]["A"] == 1
    assert "B" not in out.get("q1", {}), f"B should be filtered (maps to 0)"
    print(f"  [OK] rescale filters scores that round to 0")

def test_rescale_returns_new_dict():
    # must not mutate the input qrels
    qrels = {"q1": {"A": 2}}
    out = rescale_qrels_graded(qrels, max_score_new=5)
    assert qrels["q1"]["A"] == 2, "original qrels was mutated"
    assert out is not qrels,      "returned same object — should be a new dict"
    print(f"  [OK] rescale returns a new dict (original not mutated)")

def test_rescale_invalid_max_score_orig():
    # max_score_orig <= 0 must raise
    try:
        rescale_qrels_graded({"q1": {"A": 2}}, max_score_new=5, max_score_orig=0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print(f"  [OK] rescale raises ValueError for max_score_orig=0")

def test_rescale_relative_order_preserved():
    # relative order of PMIDs must not change after rescaling
    qrels = {"q1": {"A": 2, "B": 1}}
    for target_max in [3, 5, 7, 10]:
        out = rescale_qrels_graded(qrels, max_score_new=target_max)
        assert out["q1"]["A"] > out["q1"]["B"], (
            f"Order inverted at target_max={target_max}: A={out['q1']['A']}, B={out['q1']['B']}"
        )
    print(f"  [OK] rescale preserves relative score ordering for targets 3, 5, 7, 10")


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("qrels_builder tests")
    print("=" * 60)

    print("\n-- Saved file structure tests (fast, no rebuild) --")
    test_saved_structure()
    test_graded_structure()
    test_graded_superset_of_binary()
    test_binary_derivable_from_graded()
    test_no_empty_topics()
    test_all_65_topics_present()
    test_score_distribution()

    print("\n-- Synthetic unit test (no file I/O) --")
    test_synthetic_build()
    test_graded_score_mapping()

    print("\n-- rescale_qrels_graded tests (synthetic, no file I/O) --")
    test_rescale_0_2_to_0_5()
    test_rescale_0_2_to_0_7()
    test_rescale_identity()
    test_rescale_preserves_topic_structure()
    test_rescale_filters_zeroed_scores()
    test_rescale_returns_new_dict()
    test_rescale_invalid_max_score_orig()
    test_rescale_relative_order_preserved()

    print("\n-- Rebuild tests (slow: re-reads submissions.json) --")
    print("  Loading corpus PMIDs for filter check...")
    corpus_pmids = _load_corpus_pmids()
    print(f"  Corpus size: {len(corpus_pmids)} PMIDs")
    test_corpus_membership()
    test_rebuild_binary_matches_saved(corpus_pmids)
    test_rebuild_graded_matches_saved(corpus_pmids)

    print("\n" + "=" * 60)
    print("All qrels_builder tests passed.")
    print("=" * 60)
