"""
src/data/query_builder.py

Converts a BioGen topic dict into a query string for a given field.

Public API:
    build_query(topic, field) -> str
"""


def build_query(topic: dict, field: str) -> str:
    """
    Return a query string from a BioGen topic dict.

    Args:
        topic: dict with keys 'topic', 'question', 'narrative'.
        field: one of:
            'topic'           -> short keyword label (3-8 words)
            'question'        -> full clinical question (10-20 words)
            'narrative'       -> extended relevance description (30-60 words)
            'topic+question'  -> topic + question (no narrative)
            'topic+narrative' -> topic + narrative (no question)
            'concatenated'    -> topic + question + narrative (all three)

    Raises:
        ValueError: if field is not one of the six recognised values.
    """
    if field == "topic":
        return topic["topic"]
    if field == "question":
        return topic["question"]
    if field == "narrative":
        return topic["narrative"]
    if field == "topic+question":
        return f"{topic['topic']} {topic['question']}"
    if field == "topic+narrative":
        return f"{topic['topic']} {topic['narrative']}"
    if field == "concatenated":
        return f"{topic['topic']} {topic['question']} {topic['narrative']}"
    raise ValueError(
        f"Unknown query field: '{field}'. "
        "Use 'topic', 'question', 'narrative', "
        "'topic+question', 'topic+narrative', or 'concatenated'."
    )


# ---------------------------------------------------------------------------
# Self-test: python -m src.data.query_builder
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _t = {
        "id": "1",
        "topic": "sleep apnea treatment",
        "question": "What is the best treatment for obstructive sleep apnea in adults?",
        "narrative": "Relevant documents discuss CPAP, surgical options and lifestyle changes.",
    }

    assert build_query(_t, "topic") == "sleep apnea treatment"
    assert build_query(_t, "question").startswith("What is")
    assert build_query(_t, "narrative").startswith("Relevant")
    assert build_query(_t, "topic+question").startswith("sleep apnea treatment What")
    assert build_query(_t, "topic+narrative").startswith("sleep apnea treatment Relevant")
    assert build_query(_t, "concatenated").count(" ") > 5

    try:
        build_query(_t, "bad_field")
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "bad_field" in str(e)

    print("[ok] query_builder self-test passed")
