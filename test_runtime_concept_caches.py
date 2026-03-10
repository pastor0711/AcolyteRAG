from pathlib import Path
import sys

_PACKAGE_ROOT = str(Path(__file__).resolve().parent.parent)
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)

import acolyterag.concept_expansion as concept_expansion
import acolyterag.concepts as concepts
import acolyterag.narrative_elements as narrative_elements
import acolyterag.retrieval as retrieval
from acolyterag.text_processing import _tokenize


def test_register_concepts_invalidates_expand_concepts_cache():
    group_name = "cache_test_group"
    token = "cachetoken"

    concepts.unregister_concepts(group_name)
    try:
        tokens = (token,)
        before = concept_expansion._expand_concepts(tokens)

        concepts.register_concepts(group_name, [token])
        after = concept_expansion._expand_concepts(tokens)

        assert group_name not in before
        assert group_name in after
    finally:
        concepts.unregister_concepts(group_name)


def test_register_concepts_invalidates_narrative_elements_cache():
    token = "ragecache"
    text = f"{token} again."

    concepts._registry["anger"].discard(token)
    concepts._clear_runtime_concept_caches()
    try:
        before = narrative_elements._extract_narrative_elements(text)

        concepts.register_concepts("anger", [token])
        after = narrative_elements._extract_narrative_elements(text)

        assert "anger" not in before["emotions"]
        assert "anger" in after["emotions"]
    finally:
        concepts._registry["anger"].discard(token)
        concepts._clear_runtime_concept_caches()


def test_register_concepts_invalidates_tokenize_cache_for_dynamic_terms():
    group_name = "cache_tokenize_group"
    try:
        concepts.unregister_concepts(group_name)

        before = _tokenize("during step by step")

        concepts.register_concepts(group_name, ["during", "step by step"])
        after = _tokenize("during step by step")

        assert "during" not in before
        assert "step by step" not in before
        assert "during" in after
        assert "step by step" in after
        assert group_name in concept_expansion._expand_concepts(after)
    finally:
        concepts.unregister_concepts(group_name)


def test_unregister_concepts_invalidates_expand_concepts_cache():
    group_name = "cache_unregister_group"
    token = "cacheunset"

    concepts.unregister_concepts(group_name)
    try:
        concepts.register_concepts(group_name, [token])

        tokens = (token,)
        before = concept_expansion._expand_concepts(tokens)

        assert group_name in before

        concepts.unregister_concepts(group_name)
        after = concept_expansion._expand_concepts(tokens)

        assert group_name not in after
    finally:
        concepts.unregister_concepts(group_name)


def test_register_concepts_does_not_break_retrieval():
    group_name = "cache_smoke_group"
    token = "retrievalcache"
    history = [
        {"role": "user", "content": "Earlier we discussed retrievalcache in detail."},
        {"role": "assistant", "content": "Yes, retrievalcache was a major theme."},
        {"role": "user", "content": "Can you remind me what mattered there?"},
        {"role": "assistant", "content": "It mattered because it was memorable."},
        {"role": "user", "content": "What about something else?"},
        {"role": "assistant", "content": "Something else was less relevant."},
        {"role": "user", "content": "Please retrieve that memory."},
    ]

    concepts.unregister_concepts(group_name)
    try:
        concepts.register_concepts(group_name, [token])

        results = retrieval.retrieve_related_messages(
            history,
            query_text=token,
            max_retrieved=1,
            exclude_last_n=1,
        )

        assert len(results) == 1
        assert "[RELATED_MEMORY]" in results[0]["content"]
    finally:
        concepts.unregister_concepts(group_name)
