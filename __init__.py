"""
AcolyteRAG: Open Retrieval-Augmented Generation

This package provides tools for analyzing conversation history, extracting
narrative elements/concepts, and retrieving highly relevant past messages.
It is designed to give AI characters a human-like memory system.
"""
from .retrieval import (
    retrieve_related_messages,
    retrieve_related_messages_async,
    _retrieve_related_messages_sync,
)
from .analysis import (
    summarize_conversation_window,
    extract_conversation_topics,
    get_memory_statistics,
    create_memory_index,
)
from .concepts import register_concepts, unregister_concepts, list_concept_groups

__all__ = [
    "retrieve_related_messages",
    "retrieve_related_messages_async",
    "summarize_conversation_window",
    "extract_conversation_topics",
    "get_memory_statistics",
    "create_memory_index",
    "register_concepts",
    "unregister_concepts",
    "list_concept_groups",
]
