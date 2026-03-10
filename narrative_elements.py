"""
Extracts high-level narrative themes (emotions, actions, locations) and 
named entities from text to build a structured representation of a message.
"""
from __future__ import annotations
from functools import lru_cache
from typing import Dict, Iterable, Set, Tuple

from .constants import _NAME_PATTERN
from .text_processing import _coerce_text, _tokenize
from .concepts import get_word_to_groups


_SCORING_GROUPS: Dict[str, Set[str]] = {
    'emotions'          : {'anger', 'betrayal', 'fear', 'guilt', 'happiness', 'love', 'sadness', 'surprise', 'trust'},
    'actions'           : {'conflict', 'create', 'destroy', 'help', 'investigate', 'meeting', 'travel'},
    'locations'         : {'home', 'outdoors', 'school', 'social_venue', 'workplace'},
}

_COMMON_SENTENCE_STARTERS = {
    "The", "A", "An", "Some", "Any", "Each", "Every", "Both", "All", "No",
    "In", "On", "At", "To", "By", "For", "Of", "Up", "As", "Into",
    "With", "From", "About", "Over", "After", "Before", "Under",
    "Between", "Through", "During", "Against", "Without",
    "And", "But", "Or", "So", "Yet", "Nor", "Although", "Because",
    "If", "When", "While", "Though", "Unless", "Until", "Since",
    "He", "She", "It", "We", "They", "I", "You",
    "His", "Her", "Its", "My", "Our", "Your", "Their",
    "Him", "Them", "Us", "Me",
    "Is", "Are", "Was", "Were", "Be", "Been", "Am",
    "Do", "Does", "Did", "Have", "Has", "Had",
    "Will", "Would", "Should", "Could", "Can", "May", "Might", "Shall", "Must",
    "What", "When", "Where", "Why", "Who", "Which", "How", "Whose", "Whom",
    "Please", "Let", "Just", "Now", "Then", "Here", "There",
    "Yes", "No", "Not", "Also", "Well", "Ok", "Okay",
    "This", "That", "These", "Those",
    "Thanks", "Sorry", "Hey", "Hi", "Hello",
}

_COMMON_SENTENCE_IMPERATIVES = {
    "compare",
    "describe",
    "draft",
    "explain",
    "list",
    "outline",
    "review",
    "show",
    "summarize",
    "tell",
}

_TRAILING_SENTENCE_END_CLOSERS = "\"')]}"


def _is_sentence_start_match(text: str, start: int) -> bool:
    prefix = text[:start].rstrip()
    if not prefix:
        return True

    idx = len(prefix) - 1
    while idx >= 0 and prefix[idx] in _TRAILING_SENTENCE_END_CLOSERS:
        idx -= 1
    return idx >= 0 and prefix[idx] in ".!?"


def _should_skip_entity_candidate(text: str, start: int, entity: str) -> bool:
    if not _is_sentence_start_match(text, start):
        return False
    if entity in _COMMON_SENTENCE_STARTERS:
        return True
    return entity.lower() in _COMMON_SENTENCE_IMPERATIVES


@lru_cache(maxsize=2048)
def _extract_narrative_elements(text: str) -> Dict[str, Tuple[str, ...]]:
    """
    Parses a text string to extract narrative dimensions matching the _SCORING_GROUPS 
    and uses a regex to pull out capitalized Named Entities (ignoring common caps).
    """
    return _extract_narrative_elements_with_groups(text, _SCORING_GROUPS)


def _extract_narrative_elements_with_groups(
    text: str,
    scoring_groups: Dict[str, Iterable[str]],
) -> Dict[str, Tuple[str, ...]]:
    safe_text = _coerce_text(text)
    tokens = _tokenize(safe_text)
    word_to_groups = get_word_to_groups()

    matched_groups: set[str] = set()
    for token in tokens:
        if token in word_to_groups:
            matched_groups.update(word_to_groups[token])

    result: Dict[str, Tuple[str, ...]] = {}
    for dim, group_set in scoring_groups.items():
        result[dim] = tuple(sorted(matched_groups & set(group_set)))

    entities_found: set[str] = set()
    for match in _NAME_PATTERN.finditer(safe_text):
        entity = match.group(0)
        if _should_skip_entity_candidate(safe_text, match.start(), entity):
            continue
        entities_found.add(entity)
    entities = tuple(sorted(entities_found))
    result["entities"] = entities

    return result
