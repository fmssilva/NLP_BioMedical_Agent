import json
from pathlib import Path


# Load corpus documents from the JSONL file (one JSON object per line).
# size=None -> load all; size=N -> load first N docs (fast local testing)
def load_corpus(path: str | Path, size: int | None = None) -> list[dict]:
    corpus = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                corpus.append(json.loads(line))
                if size is not None and len(corpus) >= size:
                    break
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



