from pathlib import Path
import sys

_PACKAGE_ROOT = str(Path(__file__).resolve().parent.parent)
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)

import acolyterag.concepts as concepts
from acolyterag.analysis import (
    create_memory_index,
    extract_conversation_topics,
    get_memory_statistics,
    summarize_conversation_window,
)
from acolyterag.concept_expansion import _expand_concepts
from acolyterag.narrative_elements import _extract_narrative_elements
from acolyterag.text_processing import _canonicalize, _normalize, _tokenize


def test_normalize_and_tokenize_strip_think_blocks_and_punctuation():
    text = "<think>ignore this</think> Running, TALKED! plans."

    assert _normalize(text) == "running talked plans"
    assert _tokenize(text) == ("run", "talk", "plan")


def test_canonicalize_uses_irregular_and_suffix_rules():
    assert _canonicalize("running") == "run"
    assert _canonicalize("talked") == "talk"
    assert _canonicalize("happiness") == "happy"


def test_tokenize_preserves_registered_terms_and_stops_mangling_common_words():
    assert _tokenize("boss before forest make") == ("boss", "before", "forest", "make")
    assert _tokenize("boss class discuss stress") == ("boss", "class", "discuss", "stress")
    assert "authority" in _expand_concepts(_tokenize("boss"))
    assert "past" in _expand_concepts(_tokenize("before"))


def test_tokenize_registered_phrase_consumes_component_words():
    group_name = "phrase_tokenize_group"

    concepts.unregister_concepts(group_name)
    try:
        concepts.register_concepts(group_name, ["step by step"])

        assert _tokenize("step by step") == ("step by step",)
        assert _tokenize("learn step by step today") == ("learn", "step by step", "today")
        assert _tokenize("step by step and step by step") == ("step by step", "step by step")
    finally:
        concepts.unregister_concepts(group_name)


def test_extract_narrative_elements_identifies_dimensions_and_entities():
    elements = _extract_narrative_elements("Alice felt angry after the meet at the office with Bob.")

    assert elements["emotions"] == ("anger",)
    assert elements["actions"] == ("meeting",)
    assert elements["locations"] == ("workplace",)
    assert elements["entities"] == ("Alice", "Bob")


def test_extract_narrative_elements_ignores_sentence_start_imperatives():
    elements = _extract_narrative_elements("Tell Alice to review Python with Bob.")

    assert elements["entities"] == ("Alice", "Bob", "Python")


def test_summarize_conversation_window_includes_topics_mentions_and_notable_message():
    history = [
        {"role": "user", "content": "Alice asked about Python testing."},
        {"role": "assistant", "content": "Python testing helps Alice verify regressions."},
        {"role": "user", "content": "Can Python testing catch production regressions for Alice?"},
    ]

    summary = summarize_conversation_window(history)

    assert "Key topics:" in summary
    assert "python" in summary
    assert "test" in summary
    assert "Mentions:" in summary
    assert "Alice" in summary
    assert "Notable:" in summary


def test_summarize_conversation_window_ignores_sentence_start_imperatives_in_mentions():
    history = [
        {"role": "user", "content": "Tell me about Python."},
        {
            "role": "assistant",
            "content": "Django and Python libraries support production services across engineering teams.",
        },
    ]

    summary = summarize_conversation_window(history)

    assert "Mentions: Django, Python" in summary
    assert "Tell" not in summary


def test_summarize_conversation_window_handles_missing_content():
    assert summarize_conversation_window([{"role": "user"}]) == ""


def test_extract_conversation_topics_respects_min_frequency():
    history = [
        {"role": "user", "content": "Python testing matters."},
        {"role": "assistant", "content": "Python testing catches bugs."},
        {"role": "user", "content": "Python helps."},
    ]

    topics = extract_conversation_topics(history, min_topic_frequency=2)

    assert topics["python"] == 3
    assert topics["test"] == 2
    assert "bug" not in topics


def test_get_memory_statistics_handles_empty_history():
    stats = get_memory_statistics([])

    assert stats == {
        "total_messages": 0,
        "user_messages": 0,
        "assistant_messages": 0,
        "entities": [],
        "average_importance": 0.0,
        "top_tokens": {},
    }


def test_get_memory_statistics_counts_roles_entities_and_top_tokens():
    history = [
        {"role": "user", "content": "Alice asked about Python."},
        {"role": "assistant", "content": "Python helps Alice ship software."},
        {"role": "user", "content": "Bob also uses Python."},
    ]

    stats = get_memory_statistics(history)

    assert stats["total_messages"] == 3
    assert stats["user_messages"] == 2
    assert stats["assistant_messages"] == 1
    assert stats["entities"] == ["Alice", "Bob", "Python"]
    assert stats["top_tokens"]["python"] == 3


def test_get_memory_statistics_ignores_sentence_start_imperatives():
    history = [
        {"role": "user", "content": "Tell me about Python."},
        {"role": "assistant", "content": "List Django libraries."},
    ]

    stats = get_memory_statistics(history)

    assert stats["entities"] == ["Django", "Python"]


def test_create_memory_index_splits_history_into_chunks_with_metadata():
    history = [
        {"role": "user", "content": "Alice discussed Python testing."},
        {"role": "assistant", "content": "Python testing helps Alice."},
        {"role": "user", "content": "Bob reviewed deployment steps."},
    ]

    index = create_memory_index(history, chunk_size=2)

    assert len(index) == 2
    assert index[0]["chunk_start"] == 0
    assert index[0]["chunk_end"] == 2
    assert "Alice" in index[0]["entities"]
    assert "python" in index[0]["top_tokens"]
    assert index[1]["chunk_start"] == 2
    assert index[1]["chunk_end"] == 3
    assert index[1]["entities"] == ["Bob"]
