import json
from pathlib import Path
from collections import Counter

from src.data.loader import load_corpus


######################################################################
## Qrels builder — parse biogen_2024_submissions.json into graded
## and binary relevance judgements.
######################################################################


# Default graded relevance mapping — 0–5 scale (project default).
# supporting=5, neutral=2 gives better NDCG separation than 0–2:
_DEFAULT_GRADED_SCORE = {
    "supporting":       5,  # strong evidence — full relevance
    "neutral":          2,  # mentioned but not clearly supporting — partial relevance
    "not relevant":     0,
    "contradicting":    0,
    "invalid citation": 0,
}
# Binary view = graded filtered at this threshold.
# score >= 5 means "supporting only" — only strong evidence counts as binary-relevant.
# Use threshold=2 to include neutral docs in the binary view.
_DEFAULT_BINARY_THRESHOLD = 5

# Default paths relative to project root for local testing 
_ROOT = Path(__file__).resolve().parents[2]


def build_qrels_graded(
    submissions_path: str | Path,
    corpus_pmids: set | None = None,
    graded_score: dict | None = None,
) -> dict:
    """
    Build graded qrels from biogen_2024_submissions.json.
    For each (topic, PMID) pair:
        - score = MAX graded score across all citations/systems.
        - Max wins: one strong citation beats many weak ones.
    Default scale (0–5): supporting=5, neutral=2, others=0.
    Only stores PMIDs with score >= 1 (0-score = irrelevant; standard IR practice:
        not listed in qrels = assumed non-relevant for trec_eval / ranx).
    """
    if graded_score is None:
        graded_score = _DEFAULT_GRADED_SCORE

    with open(submissions_path, encoding="utf-8") as f:
        submissions = json.load(f)

    skipped_oor = []  # just to check PMIDs that are cited but not in corpus
    qrels = {}

    for entry in submissions:
        qid = entry["question_id"]
        # track the best score seen for each PMID for this topic
        pmid_scores: dict[str, int] = {}

        for system_answer in entry["machine_generated_answers"].values():
            for sentence in system_answer.get("answer_sentences", []):
                citations = sentence.get("citation_assessment")
                if not citations:
                    continue
                for citation in citations:
                    rel  = citation.get("evidence_relation", "")
                    pmid = citation["cited_pmid"]

                    if corpus_pmids is not None and pmid not in corpus_pmids:
                        skipped_oor.append((qid, pmid))
                        continue

                    # keep highest score seen for this PMID for this topic across all citations/systems
                    score = graded_score.get(rel, 0)
                    if score > pmid_scores.get(pmid, 0):
                        pmid_scores[pmid] = score

        # only keep PMIDs with score >= 1 (supporting or neutral)
        graded_pmids = {pmid: score for pmid, score in pmid_scores.items() if score >= 1}
        if graded_pmids:
            qrels[qid] = graded_pmids

    if skipped_oor:
        # deduplicate — same PMID can appear many times across systems
        unique_oor = list(dict.fromkeys(skipped_oor))
        print(f"  Note: {len(unique_oor)} unique (topic, PMID) pairs not in corpus — skipped:")
        for qid, pmid in unique_oor[:5]:  # first 5 only
            print(f"    topic {qid} -> PMID {pmid}")
        if len(unique_oor) > 5:
            print(f"    ... and {len(unique_oor) - 5} more")

    return qrels


def build_qrels(
    submissions_path: str | Path,
    corpus_pmids: set | None = None,
    graded_score: dict | None = None,
    binary_threshold: int = _DEFAULT_BINARY_THRESHOLD,
) -> dict:
    """
    Binary qrels derived from graded: filter score >= binary_threshold.
    Returns {topic_id: {pmid: 1}} for all PMIDs with score >= threshold.
    """
    graded = build_qrels_graded(
        submissions_path,
        corpus_pmids=corpus_pmids,
        graded_score=graded_score,
    )
    return {
        qid: {pmid: 1 for pmid, s in docs.items() if s >= binary_threshold}
        for qid, docs in graded.items()
        if any(s >= binary_threshold for s in docs.values())
    }


def rescale_qrels_graded(
    qrels_graded: dict,
    max_score_new: int,
    max_score_orig: int = 5,
) -> dict:
    """
    Linearly rescale graded qrels from the original 0–max_score_orig scale
    to a new 0–max_score_new scale (rounding to nearest integer).
    Args:
        qrels_graded:   {topic_id: {pmid: int_score}}  — standard output of build_qrels_graded.
        max_score_new:  target maximum score (e.g. 2 or 7).
        max_score_orig: current maximum score (default 5, matching the project GRADED_SCORE).
    Returns: New qrels dict with rescaled integer scores.
    """
    if max_score_orig <= 0:
        raise ValueError("max_score_orig must be > 0")
    factor = max_score_new / max_score_orig

    rescaled: dict = {}
    for qid, docs in qrels_graded.items():
        new_docs = {}
        for pmid, score in docs.items():
            new_score = round(score * factor)
            if new_score >= 1:
                new_docs[pmid] = new_score
        if new_docs:
            rescaled[qid] = new_docs
    return rescaled



def save_qrels(qrels: dict, output_path: str | Path) -> None:
    """
    Save qrels dict to disk as JSON.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(qrels, f, indent=2)
    print(f"qrels saved to {output_path}  ({len(qrels)} topics)")


def print_qrels_summary(
    qrels_graded: dict,
    qrels_binary: dict,
    submissions_path: str | Path,
    binary_threshold: int = _DEFAULT_BINARY_THRESHOLD,
) -> None:
    """
    Print a summary of the built qrels.
    Shows raw citation distribution, score breakdown, per-topic stats, sample topic.
    """
    # -- raw citation distribution (all systems/topics, before corpus filter) --
    rel_counter: Counter = Counter()
    with open(submissions_path, encoding="utf-8") as f:
        submissions = json.load(f)
    for entry in submissions:
        for sys_ans in entry["machine_generated_answers"].values():
            for sent in sys_ans.get("answer_sentences", []):
                for cit in sent.get("citation_assessment") or []:
                    rel_counter[cit.get("evidence_relation", "unknown")] += 1
    total_cit = sum(rel_counter.values())

    print(f"\n{'='*60}")
    print("Total citations (all systems, all topics)")
    print(f"{'='*60}")
    for rel, cnt in rel_counter.most_common():
        print(f"  {rel:<25} : {cnt:>6}  ({cnt/total_cit*100:.1f}%)")
    print(f"  {'TOTAL':<25} : {total_cit:>6}")

    # -- graded score distribution --
    score_cnt: Counter = Counter()
    for docs in qrels_graded.values():
        for s in docs.values():
            score_cnt[s] += 1
    total_graded = sum(score_cnt.values())
    total_binary = sum(len(v) for v in qrels_binary.values())

    print(f"\n{'='*60}")
    print("Qrels score distribution (corpus-filtered, unique per topic)")
    print(f"{'='*60}")
    print(f"  score=5 (supporting)  : {score_cnt[5]:>5}  ({score_cnt[5]/total_graded*100:.1f}%)")
    print(f"  score=2 (neutral)     : {score_cnt[2]:>5}  ({score_cnt[2]/total_graded*100:.1f}%)")
    print(f"  total graded PMIDs    : {total_graded:>5}")
    print(f"  total binary PMIDs    : {total_binary:>5}  (score >= {binary_threshold} = supporting only)")

    # -- per-topic stats --
    per_s5 = [sum(1 for s in d.values() if s == 5) for d in qrels_graded.values()]
    per_s2 = [sum(1 for s in d.values() if s == 2) for d in qrels_graded.values()]
    no_neutral = sum(1 for x in per_s2 if x == 0)

    print(f"\n{'='*60}")
    print("Per-topic statistics")
    print(f"{'='*60}")
    print(f"  avg relevant/topic (binary)    : {total_binary/len(qrels_binary):.1f}")
    print(f"  supporting (score=5)/topic      : avg={sum(per_s5)/len(per_s5):.1f}  min={min(per_s5)}  max={max(per_s5)}")
    print(f"  neutral    (score=2)/topic      : avg={sum(per_s2)/len(per_s2):.1f}  min={min(per_s2)}  max={max(per_s2)}")
    print(f"  topics with no neutral entries  : {no_neutral} / {len(qrels_graded)}")

    # -- sample topic (one with both score=5 and score=2 entries) --
    sample_qid = next(
        (qid for qid, docs in qrels_graded.items()
         if any(s == 2 for s in docs.values()) and any(s == 5 for s in docs.values())),
        next(iter(qrels_graded))
    )
    sample = qrels_graded[sample_qid]
    s5_pmids = [p for p, s in sample.items() if s == 5]
    s2_pmids = [p for p, s in sample.items() if s == 2]

    print(f"\n{'='*60}")
    print(f"Sample topic: {sample_qid}  ({len(sample)} graded PMIDs total)")
    print(f"{'='*60}")
    print(f"  score=5 (supporting): {s5_pmids[:4]}{'...' if len(s5_pmids) > 4 else ''}  [{len(s5_pmids)} total]")
    print(f"  score=2 (neutral)   : {s2_pmids[:4]}{'...' if len(s2_pmids) > 4 else ''}  [{len(s2_pmids)} total]")
    print(f"  (score=0 not stored — trec_eval assumes unlisted = non-relevant)")




def run_qrels_builder(
    submissions_path: str | Path = _ROOT / "data" / "biogen_2024_submissions.json",
    corpus_path:      str | Path = _ROOT / "data" / "filtered_pubmed_abstracts.txt",
    output_binary:    str | Path = _ROOT / "results" / "qrels.json",
    output_graded:    str | Path = _ROOT / "results" / "qrels_graded.json",
    graded_score:     dict | None = None,
    binary_threshold: int = _DEFAULT_BINARY_THRESHOLD,
) -> tuple[dict, dict]:
    """
    Orchestrates functions above: 
        - Build and save both binary and graded qrels. Always overwrites existing files.
    """
    output_binary = Path(output_binary)
    output_graded = Path(output_graded)

    print("Loading corpus for PMID filter...")
    corpus = load_corpus(corpus_path)
    corpus_pmids = {doc["id"] for doc in corpus}
    print(f"  Corpus size: {len(corpus_pmids)} PMIDs")

    # single parse: graded is the source of truth
    print("\nBuilding graded qrels (single parse)...")
    qrels_graded = build_qrels_graded(
        submissions_path,
        corpus_pmids=corpus_pmids,
        graded_score=graded_score,
    )
    save_qrels(qrels_graded, output_graded)

    # derive binary from graded — no second file read needed
    print(f"\nDeriving binary qrels (score >= {binary_threshold})...")
    qrels = {
        qid: {pmid: 1 for pmid, s in docs.items() if s >= binary_threshold}
        for qid, docs in qrels_graded.items()
        if any(s >= binary_threshold for s in docs.values())
    }
    save_qrels(qrels, output_binary)

    print_qrels_summary(qrels_graded, qrels, submissions_path, binary_threshold=binary_threshold)

    return qrels, qrels_graded


#################################################################
##                  LOCAL TEST                                 ##
#################################################################
if __name__ == "__main__":
    run_qrels_builder()
