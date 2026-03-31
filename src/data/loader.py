import json
from pathlib import Path


# Load all 4194 corpus documents from the JSONL file (one JSON object per line).
def load_corpus(path: str | Path) -> list[dict]:
    corpus = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                corpus.append(json.loads(line))
    return corpus


# Load the 65 BioGen topics from the JSON file.
def load_topics(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # topics JSON wraps the list under a "topics" key
    if isinstance(data, dict) and "topics" in data:
        return data["topics"]
    # handle flat list just in case
    return data


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]

    corpus_path = root / "data" / "filtered_pubmed_abstracts.txt"
    topics_path = root / "data" / "BioGen2024topics.json"

    print("Loading corpus...")
    corpus = load_corpus(corpus_path)
    print(f"  First doc id   : {corpus[0]['id']}")
    print(f"  First doc text : {corpus[0]['contents'][:80]}...")
    assert len(corpus) == 4194, f"Expected 4194 docs, got {len(corpus)}"
    assert all("id" in d and "contents" in d for d in corpus), "Some docs missing 'id' or 'contents'"
    assert all(d["contents"].strip() for d in corpus), "Some docs have empty contents"
    print(f"  corpus size: {len(corpus)} docs [OK]")

    print("Loading topics...")
    topics = load_topics(topics_path)
    print(f"  First topic id    : {topics[0]['id']}")
    print(f"  First topic text  : {topics[0]['topic']}")
    assert len(topics) == 65, f"Expected 65 topics, got {len(topics)}"
    ids = [t["id"] for t in topics]
    assert min(ids) == 116 and max(ids) == 180, f"Topic IDs range unexpected: {min(ids)}-{max(ids)}"
    print(f"  topics size: {len(topics)}, IDs {min(ids)}-{max(ids)} [OK]")

    print("All loader tests passed.")
