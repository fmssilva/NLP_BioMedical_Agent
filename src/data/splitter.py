import json
from pathlib import Path

from src.data.loader import load_topics

# Default paths relative to project root. Can be overridden by args to run_splitter().
_ROOT = Path(__file__).resolve().parents[2]


def split_queries(topics: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Split topics into train (odd IDs) and test (even IDs). 
    """
    train = [t for t in topics if t["id"] % 2 == 1]
    test  = [t for t in topics if t["id"] % 2 == 0]
    return train, test


def save_splits(train: list[dict], test: list[dict], output_dir: str | Path) -> None:
    """
    Save train/test splits to disk as JSON.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "train_queries.json", "w", encoding="utf-8") as f:
        json.dump(train, f, indent=2)
    with open(output_dir / "test_queries.json", "w", encoding="utf-8") as f:
        json.dump(test, f, indent=2)
    print(f"Splits saved to {output_dir}")



def run_splitter(
    topics_path: str | Path = _ROOT / "data" / "BioGen2024topics.json",
    splits_dir:  str | Path = _ROOT / "results" / "splits",
) -> tuple[list[dict], list[dict]]:
    """
    Build train/test splits from topics and save to disk.
    """
    
    splits_dir = Path(splits_dir)

    topics = load_topics(topics_path)
    train, test = split_queries(topics)

    train_ids = {t["id"] for t in train}
    test_ids  = {t["id"] for t in test}

    assert not (train_ids & test_ids), f"Overlap between train and test: {train_ids & test_ids}"
    assert all(i % 2 == 1 for i in train_ids), "Train set contains even IDs"
    assert all(i % 2 == 0 for i in test_ids),  "Test set contains odd IDs"
    all_ids = {t["id"] for t in topics}
    assert train_ids | test_ids == all_ids, "Some topics are missing from both splits"

    print(f"Train: {len(train)} queries  IDs: {sorted(train_ids)[:5]}...")
    print(f"Test : {len(test)} queries   IDs: {sorted(test_ids)[:5]}...")

    save_splits(train, test, splits_dir)
    print("Splits saved.")
    return train, test


if __name__ == "__main__":
    train, test = run_splitter()
    print("All splitter tests passed.")
