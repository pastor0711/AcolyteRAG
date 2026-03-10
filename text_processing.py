"""
Core text utilities for cleaning, tokenization, stemming, and lemmatization.
Used to preprocess queries and candidate documents before scoring.
"""
from __future__ import annotations
import re
from functools import lru_cache
from typing import Tuple

from .constants import (
    _STOPWORDS,
    _IRREGULAR_LEMMA_MAP,
    _PUNCTUATION_PATTERN,
    _WHITESPACE_PATTERN,
)

_THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_WORD_PATTERN = re.compile(r"\S+")


def _coerce_text(text: object) -> str:
    return text if isinstance(text, str) else ""


def _strip_think_blocks(text: str) -> str:
    return _THINK_BLOCK_PATTERN.sub("", text).strip()


@lru_cache(maxsize=2000)
def _normalize(text: str) -> str:
    text = _strip_think_blocks(_coerce_text(text))
    text = text.lower()
    text = _PUNCTUATION_PATTERN.sub(" ", text)
    text = _WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


@lru_cache(maxsize=8192)
def _canonicalize(word: str) -> str:
    if word in _IRREGULAR_LEMMA_MAP:
        return _IRREGULAR_LEMMA_MAP[word]

    if len(word) <= 3:
        return word

    if word.endswith("iness") and len(word) > 5:
        return word[:-5] + "y"

    if word.endswith("ied") and len(word) > 4:
        return word[:-3] + "y"

    if word.endswith("sses") and len(word) > 4:
        return word[:-2]

    if word.endswith(("ches", "shes", "xes", "zes", "ses", "oes")) and len(word) > 4:
        return word[:-2]

    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"

    if word.endswith("ing") and len(word) > 5:
        stem = word[:-3]
        if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
            stem = stem[:-1]
        if len(stem) >= 3:
            return stem

    if word.endswith("ed") and len(word) > 4:
        stem = word[:-2]
        if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
            stem = stem[:-1]
        if stem.endswith("i"):
            stem = stem[:-1] + "y"
        if len(stem) >= 3:
            return stem

    if word.endswith("ness") and len(word) > 6:
        stem = word[:-4]
        if stem.endswith("i"):
            stem = stem[:-1] + "y"
        if len(stem) >= 3:
            return stem

    if word.endswith("s") and len(word) > 3 and not word.endswith(("ss", "us", "is")):
        stem = word[:-1]
        if len(stem) >= 3:
            return stem

    return word


@lru_cache(maxsize=4096)
def _tokenize(text: str) -> Tuple[str, ...]:
    """
    Converts raw text into a sequence of canonicalized, non-stopword tokens for matching.
    """
    text = _normalize(text)
    if not text:
        return ()

    from .concepts import get_word_to_groups

    word_to_groups = get_word_to_groups()
    registered_phrases = sorted(
        (term for term in word_to_groups if " " in term),
        key=len,
        reverse=True,
    )

    phrase_matches: list[tuple[int, int, str]] = []
    for phrase in registered_phrases:
        for match in re.finditer(rf"(?<!\S){re.escape(phrase)}(?!\S)", text):
            start, end = match.span()
            if any(start < p_end and end > p_start for p_start, p_end, _ in phrase_matches):
                continue
            phrase_matches.append((start, end, phrase))

    phrase_matches.sort(key=lambda item: item[0])
    masked_chars = list(text)
    for start, end, _ in phrase_matches:
        masked_chars[start:end] = " " * (end - start)
    masked_text = "".join(masked_chars)

    ordered_tokens: list[tuple[int, str]] = [(start, phrase) for start, _, phrase in phrase_matches]
    for match in _WORD_PATTERN.finditer(masked_text):
        word = match.group(0)
        if len(word) <= 1:
            continue

        if word in word_to_groups:
            ordered_tokens.append((match.start(), word))
            continue

        canonical = _canonicalize(word)
        if canonical in word_to_groups:
            ordered_tokens.append((match.start(), canonical))
            continue

        if word not in _STOPWORDS and canonical not in _STOPWORDS and len(canonical) > 1:
            ordered_tokens.append((match.start(), canonical))

    ordered_tokens.sort(key=lambda item: item[0])
    return tuple(token for _, token in ordered_tokens)
