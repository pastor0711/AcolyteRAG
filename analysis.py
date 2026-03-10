"""
Tools for analyzing conversational state over a window of messages.
Provides summarization, topic extraction, and memory index generation.
"""
from __future__ import annotations
from collections import Counter
from typing import Any, Dict, List, Optional

from .text_processing import _coerce_text, _tokenize
from .narrative_elements import _extract_narrative_elements
from .importance import _calculate_importance_score
from .constants import _STOPWORDS


def summarize_conversation_window(
    history: List[Dict[str, str]],
    window_start: int = 0,
    window_end: Optional[int] = None,
    max_summary_length: int = 300,
) -> str:
    """
    Generates a concise, keyword-style summary of a slice of conversation history.
    Identifies top tokens, named entities, and the single most 'important' message.
    """
    if window_end is None:
        window_end = len(history)
    window = history[window_start:window_end]
    if not window:
        return ""

    token_counts: Counter = Counter()
    all_entities: set[str] = set()
    best_msg, best_importance = None, -1.0

    for msg in window:
        content  = _coerce_text(msg.get("content", ""))
        token_counts.update(_tokenize(content))
        elements = _extract_narrative_elements(content)
        all_entities.update(elements.get("entities", ()))
        imp = _calculate_importance_score(msg, elements)
        if imp > best_importance:
            best_importance, best_msg = imp, msg

    top_tokens = [
        tok for tok, cnt in token_counts.most_common(30)
        if cnt >= 2 and tok not in _STOPWORDS
    ][:10]

    parts: List[str] = []
    if top_tokens:
        parts.append(f"Key topics: {', '.join(top_tokens)}")
    if all_entities:
        parts.append(f"Mentions: {', '.join(sorted(all_entities)[:6])}")
    snippet = _coerce_text(best_msg.get("content", "")) if best_msg else ""
    if snippet:
        snippet = snippet[:80].replace("\n", " ")
        parts.append(f'Notable: "{snippet}…"')

    return " | ".join(parts)[:max_summary_length]


def extract_conversation_topics(
    history: List[Dict[str, str]],
    min_topic_frequency: int = 3,
) -> Dict[str, int]:
    """
    Extracts the most frequently discussed topics (tokens) across the entire history.
    """
    counter: Counter = Counter()
    for msg in history:
        counter.update(_tokenize(_coerce_text(msg.get("content", ""))))
    return {tok: cnt for tok, cnt in counter.most_common(50) if cnt >= min_topic_frequency}


def get_memory_statistics(history: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Computes global statistics for the conversation, such as message counts,
    unique entities, average importance, and most frequent tokens.
    """
    user_count = assistant_count = 0
    all_entities: set[str] = set()
    importance_sum = 0.0
    token_counter: Counter = Counter()

    for msg in history:
        role = msg.get("role", "")
        content = _coerce_text(msg.get("content", ""))
        if role == "user":
            user_count += 1
        elif role == "assistant":
            assistant_count += 1
        elements = _extract_narrative_elements(content)
        all_entities.update(elements.get("entities", ()))
        importance_sum += _calculate_importance_score(msg, elements)
        token_counter.update(_tokenize(content))

    n = len(history)
    return {
        "total_messages":     n,
        "user_messages":      user_count,
        "assistant_messages": assistant_count,
        "entities":    sorted(all_entities),
        "average_importance":     round(importance_sum / n, 4) if n else 0.0,
        "top_tokens":         {tok: cnt for tok, cnt in token_counter.most_common(10) if cnt >= 2},
    }


def create_memory_index(
    history: List[Dict[str, str]],
    chunk_size: int = 50,
) -> List[Dict[str, Any]]:
    """
    Divides the history into chunks and generates a summary and metadata index 
    for each chunk to allow efficient browsing or hierarchical retrieval.
    """
    if chunk_size < 1:
        raise ValueError(
            f"chunk_size must be a positive integer, got {chunk_size!r}."
        )

    index: List[Dict[str, Any]] = []
    for chunk_start in range(0, len(history), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(history))
        chunk     = history[chunk_start:chunk_end]

        chunk_entities: set[str] = set()
        max_imp = 0.0
        tok_counter: Counter = Counter()

        for msg in chunk:
            content  = _coerce_text(msg.get("content", ""))
            elements = _extract_narrative_elements(content)
            chunk_entities.update(elements.get("entities", ()))
            max_imp = max(max_imp, _calculate_importance_score(msg, elements))
            tok_counter.update(_tokenize(content))

        index.append({
            "chunk_start":          chunk_start,
            "chunk_end":            chunk_end,
            "summary":        summarize_conversation_window(history, chunk_start, chunk_end),
            "entities":       sorted(chunk_entities),
            "top_tokens":         [t for t, _ in tok_counter.most_common(5)],
            "max_importance": round(max_imp, 4),
        })
    return index
