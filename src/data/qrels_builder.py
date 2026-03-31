import json
from pathlib import Path

from src.data.loader import load_corpus


# Graded relevance mapping for evidence_relation values.
# Confirmed from actual data distribution: supporting(10240), neutral(1412),
# not relevant(2201), contradicting(275), invalid citation(635).
# Scale: 0-2 where 2=strong evidence, 1=weak/neutral, 0=not relevant.
GRADED_SCORE = {
    "supporting":       2,  # cited as clear evidence support
    "neutral":          1,  # mentioned but neither supports nor contradicts
    "not relevant":     0,
    "contradicting":    0,
    "invalid citation": 0,
}

# Precision threshold: score >= BINARY_THRESHOLD counts as "relevant" for P@k.
# Using 1 (i.e., at least neutral) to match the intent of graded relevance.
BINARY_THRESHOLD = 1


# Build binary qrels from biogen_2024_submissions.json.
# Any PMID cited with evidence_relation == "supporting" in any system's answer for a topic -> relevant.
# If corpus_pmids is provided, PMIDs not in the corpus are silently skipped (they can never be retrieved).
def build_qrels(submissions_path: str | Path, corpus_pmids: set | None = None) -> dict:
    with open(submissions_path, encoding="utf-8") as f:
        submissions = json.load(f)

    skipped_oor = []  # out-of-range PMIDs we drop
    qrels = {}
    for entry in submissions:
        qid = entry["question_id"]
        relevant_pmids = set()

        for system_answer in entry["machine_generated_answers"].values():
            for sentence in system_answer.get("answer_sentences", []):
                citations = sentence.get("citation_assessment")
                if not citations:
                    continue
                for citation in citations:
                    if citation.get("evidence_relation") == "supporting":
                        pmid = citation["cited_pmid"]
                        if corpus_pmids is not None and pmid not in corpus_pmids:
                            skipped_oor.append((qid, pmid))
                            continue
                        relevant_pmids.add(pmid)

        # only include topics that have at least one relevant doc
        if relevant_pmids:
            qrels[qid] = {pmid: 1 for pmid in relevant_pmids}

    if skipped_oor:
        print(f"  Note: {len(skipped_oor)} qrel PMID(s) not in corpus — skipped (cannot be retrieved):")
        for qid, pmid in skipped_oor:
            print(f"    topic {qid} -> PMID {pmid}")

    return qrels


# Build graded qrels from biogen_2024_submissions.json.
# For each (topic, PMID) pair, the score is the MAX graded score seen across all citations.
# - supporting -> 2, neutral -> 1, others -> 0
# Takes the max so that one strong citation wins over multiple weak ones.
# Only stores PMIDs with score >= 1 (filtering out purely irrelevant citations).
def build_qrels_graded(submissions_path: str | Path, corpus_pmids: set | None = None) -> dict:
    with open(submissions_path, encoding="utf-8") as f:
        submissions = json.load(f)

    skipped_oor = []
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
                    rel = citation.get("evidence_relation", "")
                    pmid = citation["cited_pmid"]

                    if corpus_pmids is not None and pmid not in corpus_pmids:
                        skipped_oor.append((qid, pmid))
                        continue

                    # get numeric score; unknown relations default to 0
                    score = GRADED_SCORE.get(rel, 0)
                    # keep the highest score seen for this PMID
                    if score > pmid_scores.get(pmid, 0):
                        pmid_scores[pmid] = score

        # only keep PMIDs with score >= 1 (supporting or neutral)
        graded_pmids = {pmid: score for pmid, score in pmid_scores.items() if score >= 1}
        if graded_pmids:
            qrels[qid] = graded_pmids

    if skipped_oor:
        # deduplicate before printing — same PMID can appear many times across systems
        unique_oor = list(dict.fromkeys(skipped_oor))
        print(f"  Note: {len(unique_oor)} unique (topic, PMID) pairs not in corpus — skipped:")
        for qid, pmid in unique_oor[:5]:  # only show first 5 to keep output clean
            print(f"    topic {qid} -> PMID {pmid}")
        if len(unique_oor) > 5:
            print(f"    ... and {len(unique_oor) - 5} more")

    return qrels


# Save qrels dict to disk as JSON.
def save_qrels(qrels: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(qrels, f, indent=2)
    print(f"qrels saved to {output_path}  ({len(qrels)} topics)")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    submissions_path = root / "data" / "biogen_2024_submissions.json"
    corpus_path      = root / "data" / "filtered_pubmed_abstracts.txt"
    output_binary    = root / "results" / "qrels.json"
    output_graded    = root / "results" / "qrels_graded.json"

    print("Loading corpus for PMID filter...")
    corpus = load_corpus(corpus_path)
    corpus_pmids = {doc["id"] for doc in corpus}
    print(f"  Corpus size: {len(corpus_pmids)} PMIDs")

    # ── Binary qrels ──────────────────────────────────────────────────────
    print("\nBuilding binary qrels...")
    qrels = build_qrels(submissions_path, corpus_pmids=corpus_pmids)
    print(f"  Topics with relevant docs : {len(qrels)}")
    all_relevant_counts = [len(v) for v in qrels.values()]
    avg = sum(all_relevant_counts) / len(all_relevant_counts)
    print(f"  Avg relevant docs/topic   : {avg:.1f}")
    print(f"  Min / Max                 : {min(all_relevant_counts)} / {max(all_relevant_counts)}")

    # spot-check: topic 116 should have some PMIDs
    sample_qid = "116"
    assert sample_qid in qrels, f"Topic {sample_qid} missing from qrels"
    print(f"  Topic {sample_qid} relevant PMIDs (first 5): {list(qrels[sample_qid].keys())[:5]}")

    # verify all qrel PMIDs are in corpus
    for qid, pmid_dict in qrels.items():
        for pmid in pmid_dict:
            assert pmid in corpus_pmids, f"PMID {pmid} in topic {qid} not in corpus (filter failed)"
    print("  All binary qrel PMIDs verified in corpus [OK]")
    save_qrels(qrels, output_binary)

    # ── Graded qrels ──────────────────────────────────────────────────────
    print("\nBuilding graded qrels...")
    qrels_graded = build_qrels_graded(submissions_path, corpus_pmids=corpus_pmids)
    print(f"  Topics with graded docs   : {len(qrels_graded)}")

    # score distribution across all (topic, PMID) pairs
    score_counts = {0: 0, 1: 0, 2: 0}
    for qid, pmid_dict in qrels_graded.items():
        for pmid, score in pmid_dict.items():
            score_counts[score] = score_counts.get(score, 0) + 1
    total_pairs = sum(score_counts.values())
    print(f"  Total graded pairs        : {total_pairs}")
    print(f"  Score=2 (supporting)      : {score_counts.get(2, 0)}")
    print(f"  Score=1 (neutral)         : {score_counts.get(1, 0)}")

    graded_counts = [len(v) for v in qrels_graded.values()]
    print(f"  Avg scored docs/topic     : {sum(graded_counts)/len(graded_counts):.1f}")
    print(f"  Min / Max                 : {min(graded_counts)} / {max(graded_counts)}")

    # graded set should be >= binary set (neutral adds more entries)
    assert len(qrels_graded) >= len(qrels), "Graded qrels should have >= topics as binary"
    print(f"  Graded topics >= binary topics [OK]  ({len(qrels_graded)} >= {len(qrels)})")

    # every binary relevant PMID should be in graded with score==2
    for qid, pmid_dict in qrels.items():
        if qid in qrels_graded:
            for pmid in pmid_dict:
                if pmid in qrels_graded[qid]:
                    assert qrels_graded[qid][pmid] == 2, (
                        f"Topic {qid} PMID {pmid}: should be score=2 in graded (was supporting in binary)"
                    )
    print("  Binary PMIDs all have score=2 in graded [OK]")

    save_qrels(qrels_graded, output_graded)
    print("\nAll qrels_builder tests passed.")
