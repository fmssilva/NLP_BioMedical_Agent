import json
from pathlib import Path

from src.data.loader import load_topics


# Split topics into train (odd IDs) and test (even IDs). Fixed forever — never change this.
def split_queries(topics: list[dict]) -> tuple[list[dict], list[dict]]:
    train = [t for t in topics if t["id"] % 2 == 1]
    test  = [t for t in topics if t["id"] % 2 == 0]
    return train, test


# Save train/test splits to disk as JSON.
def save_splits(train: list[dict], test: list[dict], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "train_queries.json", "w", encoding="utf-8") as f:
        json.dump(train, f, indent=2)
    with open(output_dir / "test_queries.json", "w", encoding="utf-8") as f:
        json.dump(test, f, indent=2)
    print(f"Splits saved to {output_dir}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    topics_path  = root / "data" / "BioGen2024topics.json"
    output_dir   = root / "results" / "splits"

    topics = load_topics(topics_path)
    train, test = split_queries(topics)

    train_ids = {t["id"] for t in train}
    test_ids  = {t["id"] for t in test}

    # no overlap
    overlap = train_ids & test_ids
    assert not overlap, f"Overlap between train and test: {overlap}"

    # correct parity
    assert all(i % 2 == 1 for i in train_ids), "Train set contains even IDs"
    assert all(i % 2 == 0 for i in test_ids),  "Test set contains odd IDs"

    # full coverage — every topic is in exactly one split
    all_ids = {t["id"] for t in topics}
    assert train_ids | test_ids == all_ids, "Some topics are missing from both splits"

    print(f"Train: {len(train)} queries  IDs: {sorted(train_ids)[:5]}...")
    print(f"Test : {len(test)} queries   IDs: {sorted(test_ids)[:5]}...")

    save_splits(train, test, output_dir)
    print("All splitter tests passed.")
