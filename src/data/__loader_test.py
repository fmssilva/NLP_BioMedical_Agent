"""
Local tests for loader.py and splitter.py.
Run with: python -m src.data.__loader_test

!!!! SET OF TESTES GENERATED WITH AI HELP - COVERS BASIC FUNCTIONALITY AND EDGE CASES FOR BOTH MODULES. 
"""

import json
import sys
from pathlib import Path

# make sure project root is on sys.path when running as __main__
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.loader import load_corpus, load_topics
from src.data.splitter import split_queries, run_splitter

CORPUS_SIZE  = 10  # fast: only load 10 docs for most tests
CORPUS_PATH  = _ROOT / "data" / "filtered_pubmed_abstracts.txt"
TOPICS_PATH  = _ROOT / "data" / "BioGen2024topics.json"
SPLITS_DIR   = _ROOT / "results" / "splits"


# ── loader.py tests ────────────────────────────────────────────────────────

def test_load_corpus_small():
    # fast: load only CORPUS_SIZE docs, check structure
    corpus = load_corpus(CORPUS_PATH, size=CORPUS_SIZE)
    assert len(corpus) == CORPUS_SIZE, f"Expected {CORPUS_SIZE}, got {len(corpus)}"
    for i, doc in enumerate(corpus):
        assert "id" in doc,       f"Doc {i} missing 'id'"
        assert "contents" in doc, f"Doc {i} missing 'contents'"
        assert isinstance(doc["id"], str),       f"Doc {i} 'id' must be str"
        assert isinstance(doc["contents"], str), f"Doc {i} 'contents' must be str"
        assert doc["contents"].strip(),          f"Doc {i} 'contents' is empty"
    print(f"  [ok]  load_corpus(size={CORPUS_SIZE}): {len(corpus)} docs, structure valid")


def test_load_corpus_size_less_than_file():
    # size=3: returns exactly 3 docs
    c3 = load_corpus(CORPUS_PATH, size=3)
    assert len(c3) == 3, f"Expected 3, got {len(c3)}"
    # same first docs as size=10
    c10 = load_corpus(CORPUS_PATH, size=10)
    for i in range(3):
        assert c3[i]["id"] == c10[i]["id"], f"Doc order mismatch at position {i}"
    print(f"  [ok]  load_corpus(size=3) order consistent with size=10")


def test_load_corpus_size_none():
    # size=None -> all 4194 docs
    corpus = load_corpus(CORPUS_PATH, size=None)
    assert len(corpus) == 4194, f"Expected 4194, got {len(corpus)}"
    assert all("id" in d and "contents" in d for d in corpus), "Some docs missing fields"
    assert all(d["contents"].strip() for d in corpus), "Some docs have empty contents"
    # IDs are all strings
    assert all(isinstance(d["id"], str) for d in corpus), "Some doc IDs not str"
    print(f"  [ok]  load_corpus(size=None): {len(corpus)} docs, all valid")


def test_load_corpus_no_duplicates():
    # PMIDs must be unique
    corpus = load_corpus(CORPUS_PATH, size=None)
    ids = [d["id"] for d in corpus]
    assert len(ids) == len(set(ids)), f"Duplicate doc IDs found: {len(ids) - len(set(ids))} dupes"
    print(f"  [ok]  no duplicate PMIDs in corpus")


def test_load_corpus_partial_consistent_with_full():
    # first CORPUS_SIZE docs from partial load match first CORPUS_SIZE of full load
    small = load_corpus(CORPUS_PATH, size=CORPUS_SIZE)
    full  = load_corpus(CORPUS_PATH, size=None)
    for i in range(CORPUS_SIZE):
        assert small[i]["id"] == full[i]["id"], f"ID mismatch at position {i}"
        assert small[i]["contents"] == full[i]["contents"], f"Contents mismatch at position {i}"
    print(f"  [ok]  size={CORPUS_SIZE} slice matches beginning of full corpus")


def test_load_topics():
    # 65 topics with IDs 116-180, required fields present
    topics = load_topics(TOPICS_PATH)
    assert len(topics) == 65, f"Expected 65 topics, got {len(topics)}"
    ids = [t["id"] for t in topics]
    assert min(ids) == 116 and max(ids) == 180, f"ID range wrong: {min(ids)}-{max(ids)}"
    for i, t in enumerate(topics):
        assert "id" in t,    f"Topic {i} missing 'id'"
        assert "topic" in t, f"Topic {i} missing 'topic'"
        assert isinstance(t["id"], int), f"Topic {i} 'id' must be int, got {type(t['id'])}"
        assert isinstance(t["topic"], str) and t["topic"].strip(), f"Topic {i} 'topic' empty"
    print(f"  [ok]  load_topics: {len(topics)} topics, IDs {min(ids)}-{max(ids)}, structure valid")


def test_topics_have_expected_fields():
    # 'question' and 'narrative' should be present (used in BEST_QUERY_FIELD=concatenated)
    topics = load_topics(TOPICS_PATH)
    for t in topics:
        assert "question" in t or "narrative" in t, (
            f"Topic {t['id']} missing 'question'/'narrative' — needed for concatenated field"
        )
    print(f"  [ok]  all topics have 'question' or 'narrative' field")


# ── splitter.py tests ──────────────────────────────────────────────────────

def test_split_queries_odd_even():
    # split_queries must use odd -> train, even -> test
    topics = load_topics(TOPICS_PATH)
    train, test = split_queries(topics)
    assert all(t["id"] % 2 == 1 for t in train), "Train has even-ID topics"
    assert all(t["id"] % 2 == 0 for t in test),  "Test has odd-ID topics"
    print(f"  [ok]  split_queries: train={len(train)} odd-IDs, test={len(test)} even-IDs")


def test_split_queries_coverage():
    # all topics must be in exactly one split (no overlap, no missing)
    topics = load_topics(TOPICS_PATH)
    train, test = split_queries(topics)
    train_ids = {t["id"] for t in train}
    test_ids  = {t["id"] for t in test}
    all_ids   = {t["id"] for t in topics}
    assert not (train_ids & test_ids), f"Overlap between train/test: {train_ids & test_ids}"
    assert train_ids | test_ids == all_ids, "Some topics missing from both splits"
    assert len(train) + len(test) == len(topics), "Total count doesn't match"
    print(f"  [ok]  no overlap, all {len(topics)} topics covered: {len(train)} train, {len(test)} test")


def test_run_splitter_output_files():
    # run_splitter writes JSON files and they load back correctly
    run_splitter(topics_path=TOPICS_PATH, splits_dir=SPLITS_DIR)
    train_file = SPLITS_DIR / "train_queries.json"
    test_file  = SPLITS_DIR / "test_queries.json"
    assert train_file.exists(), f"train_queries.json not found at {train_file}"
    assert test_file.exists(),  f"test_queries.json not found at {test_file}"
    with open(train_file) as f:
        train = json.load(f)
    with open(test_file) as f:
        test = json.load(f)
    assert isinstance(train, list) and isinstance(test, list), "Splits must be JSON lists"
    assert len(train) + len(test) == 65, f"Expected 65 total, got {len(train)+len(test)}"
    print(f"  [ok]  run_splitter: files exist, train={len(train)}, test={len(test)}")


def test_run_splitter_idempotent():
    # run twice -> same result (deterministic, overwrites files)
    run_splitter(topics_path=TOPICS_PATH, splits_dir=SPLITS_DIR)
    with open(SPLITS_DIR / "train_queries.json") as f:
        train1 = json.load(f)
    run_splitter(topics_path=TOPICS_PATH, splits_dir=SPLITS_DIR)
    with open(SPLITS_DIR / "train_queries.json") as f:
        train2 = json.load(f)
    ids1 = [t["id"] for t in train1]
    ids2 = [t["id"] for t in train2]
    assert ids1 == ids2, "Splitter is not idempotent — IDs differ between runs"
    print(f"  [ok]  run_splitter is idempotent: same IDs on second run")


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("loader + splitter tests  (CORPUS_SIZE=10 for fast loading)")
    print("=" * 60)

    print("\n-- loader.py --")
    test_load_corpus_small()
    test_load_corpus_size_less_than_file()
    test_load_corpus_size_none()
    test_load_corpus_no_duplicates()
    test_load_corpus_partial_consistent_with_full()
    test_load_topics()
    test_topics_have_expected_fields()

    print("\n-- splitter.py --")
    test_split_queries_odd_even()
    test_split_queries_coverage()
    test_run_splitter_output_files()
    test_run_splitter_idempotent()

    print("\n" + "=" * 60)
    print("All loader + splitter tests passed.")
    print("=" * 60)
