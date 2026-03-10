"""
Expands simple tokens into broader conceptual categories to allow 
concept-based (semantic) matching between queries and memories.
"""
from __future__ import annotations
from functools import lru_cache
from typing import FrozenSet, Tuple

from .concepts import get_word_to_groups


@lru_cache(maxsize=4096)
def _expand_concepts(tokens: Tuple[str, ...]) -> FrozenSet[str]:
    """
    Takes a list of normalized tokens and maps them to their corresponding 
    concept groups (e.g., finding the word "angry" adds the "anger" concept).
    """
    word_to_groups = get_word_to_groups()
    groups: set[str] = set()
    for token in tokens:
        if token in word_to_groups:
            groups.update(word_to_groups[token])
    return frozenset(groups)
