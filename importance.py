"""
Calculates the absolute "importance" of a message based on its narrative elements 
and content characteristics (length, emotions, actions, entities).
"""
from __future__ import annotations
from typing import Dict, Tuple

from .text_processing import _coerce_text


def _calculate_importance_score(
    message: Dict[str, str],
    elements: Dict[str, Tuple[str, ...]],
) -> float:
    """
    Computes a heuristic importance score [0.0 - 1.0] for a given message.
    
    This score helps determine which messages are intrinsically memorable, 
    even without a specific query (e.g., messages with high emotion or actions).
    """
    content = _coerce_text(message.get("content", ""))

    length_score   = min(len(content) / 500.0, 1.0) * 0.40
    emotion_score  = min(len(elements.get("emotions",  ())), 2) * 0.10
    action_score   = min(len(elements.get("actions",   ())), 2) * 0.10
    entity_score   = min(len(elements.get("entities",  ())), 3) * 0.05
    question_bonus = 0.05 if "?" in content else 0.0

    return min(length_score + emotion_score + action_score + entity_score + question_bonus, 1.0)
