"""
Converts a BioGen topic dict into a query string for a given field.
"""


def build_query(topic: dict, field: str) -> str:
    """
    Return a query string from a BioGen topic dict.
    topics is ne of the combinations bellow
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
