from pathlib import Path
import sys

_PACKAGE_ROOT = str(Path(__file__).resolve().parent.parent)
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)

from acolyterag.retrieval import retrieve_related_messages


def test_token_budget_retrieval_skips_oversized_top_candidate():
    short_match = "budget skip token"
    long_match = "budget skip token budget skip token budget skip token"
    history = [
        {"role": "assistant", "content": short_match},
        {"role": "assistant", "content": long_match},
        {"role": "user", "content": "please use token retrieval now"},
    ]

    results = retrieve_related_messages(
        history,
        query_text=short_match,
        exclude_last_n=1,
        enable_token_based_retrieval=True,
        target_token_count=5,
        current_token_count=0,
    )

    assert len(results) == 1
    assert results[0]["content"] == f"[RELATED_MEMORY] {short_match}"


def test_token_budget_retrieval_returns_selected_memories_in_chronological_order():
    older_match = "alpha beta"
    newer_better_match = "alpha beta gamma"
    history = [
        {"role": "assistant", "content": older_match},
        {"role": "assistant", "content": newer_better_match},
        {"role": "user", "content": "please use token retrieval now"},
    ]

    results = retrieve_related_messages(
        history,
        query_text=newer_better_match,
        exclude_last_n=1,
        enable_token_based_retrieval=True,
        target_token_count=10,
        current_token_count=0,
    )

    assert [result["content"] for result in results] == [
        f"[RELATED_MEMORY] {older_match}",
        f"[RELATED_MEMORY] {newer_better_match}",
    ]


def test_token_budget_retrieval_respects_max_retrieved_cap():
    history = [
        {"role": "assistant", "content": "alpha beta"},
        {"role": "assistant", "content": "alpha beta gamma"},
        {"role": "assistant", "content": "alpha beta delta"},
        {"role": "user", "content": "please use token retrieval now"},
    ]

    results = retrieve_related_messages(
        history,
        query_text="alpha beta",
        exclude_last_n=1,
        enable_token_based_retrieval=True,
        target_token_count=100,
        current_token_count=0,
        max_retrieved=1,
    )

    assert len(results) == 1
