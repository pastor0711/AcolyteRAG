"""
Contains the mathematical and heuristic functions for scoring the relevance 
of a candidate memory against a user query. Implements TF-IDF, conceptual overlap,
narrative similarity, and blended scores.
"""
from __future__ import annotations
import math
from functools import lru_cache
from typing import Dict, FrozenSet, List, Set, Tuple


def _bigrams(tokens: Tuple[str, ...]) -> FrozenSet[Tuple[str, str]]:
    return frozenset(zip(tokens, tokens[1:]))


def _build_idf(candidates_tokens: List[Tuple[str, ...]]) -> Dict[str, float]:
    N = len(candidates_tokens)
    if N == 0:
        return {}
    doc_freq: Dict[str, int] = {}
    for tokens in candidates_tokens:
        for tok in set(tokens):
            doc_freq[tok] = doc_freq.get(tok, 0) + 1
    return {tok: math.log((N + 1) / (df + 1)) + 1.0 for tok, df in doc_freq.items()}


def _idf_overlap_ratio(
    query_tokens: Tuple[str, ...],
    candidate_tokens: Tuple[str, ...],
    idf: Dict[str, float],
) -> float:
    if not query_tokens:
        return 0.0
    candidate_set = set(candidate_tokens)
    total   = sum(idf.get(t, 1.0) for t in query_tokens)
    overlap = sum(idf.get(t, 1.0) for t in query_tokens if t in candidate_set)
    return overlap / total if total else 0.0


def _tfidf_cosine(
    query_tokens: Tuple[str, ...],
    candidate_tokens: Tuple[str, ...],
    idf: Dict[str, float],
) -> float:
    if not query_tokens or not candidate_tokens:
        return 0.0

    def _tf(tokens: Tuple[str, ...]) -> Dict[str, float]:
        n = len(tokens)
        counts: Dict[str, int] = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        return {t: c / n for t, c in counts.items()}

    qtf   = _tf(query_tokens)
    ctf   = _tf(candidate_tokens)
    vocab = set(qtf) | set(ctf)
    dot   = sum(qtf.get(t, 0.0) * idf.get(t, 1.0) * ctf.get(t, 0.0) * idf.get(t, 1.0) for t in vocab)
    qmag  = math.sqrt(sum((qtf.get(t, 0.0) * idf.get(t, 1.0)) ** 2 for t in vocab))
    cmag  = math.sqrt(sum((ctf.get(t, 0.0) * idf.get(t, 1.0)) ** 2 for t in vocab))
    return dot / (qmag * cmag) if qmag and cmag else 0.0


def _jaccard(set_a: Set[str], set_b: Set[str]) -> float:
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0


def _overlap_coefficient(set_a: Set[str], set_b: Set[str]) -> float:
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / min(len(set_a), len(set_b))


def _score_fast(
    query_tokens: Tuple[str, ...],
    candidate_tokens: Tuple[str, ...],
    importance: float,
    recency: float,
) -> float:
    """
    A fast, lightweight scoring function used for an initial pass (pre-filtering candidates).
    Relies entirely on token overlap Jaccard similarity, importance, and recency.
    """
    if not query_tokens or not candidate_tokens:
        overlap = 0.0
    else:
        overlap = _jaccard(set(query_tokens), set(candidate_tokens))
    return 0.40 * overlap + 0.30 * importance + 0.30 * recency


from .text_processing import _normalize

_NARRATIVE_WEIGHTS: Dict[str, float] = {
    'emotions'          : 0.3,
    'actions'           : 0.3,
    'entities'          : 0.25,
    'locations'         : 0.2,
}

_BLEND_WEIGHTS: Dict[str, float] = {
    'tfidf'             : 0.2,
    'idf_overlap'       : 0.15,
    'concept'           : 0.2,
    'bigram'            : 0.05,
    'token_coef'        : 0.05,
    'narrative'         : 0.25,
    'substring'         : 0.1,
    'entity_bonus'      : 0.05,
    'temporal_bonus'    : 0.1,
    'action_bonus'      : 0.1,
}

_TEMPORAL_TERMS: Tuple[str, ...] = (
    "tomorrow", "yesterday", "today", "later", "earlier", "before", "after", "soon", "now",
)
_ACTION_TERMS: Tuple[str, ...] = (
    "arrives", "sent", "went", "came", "did", "happened", "occurred", "took place",
)


def _contains_any_normalized_term(normalized_text: str, terms: Tuple[str, ...]) -> bool:
    if not normalized_text:
        return False

    padded_text = f" {normalized_text} "
    for term in terms:
        term_norm = _normalize(term)
        if term_norm and f" {term_norm} " in padded_text:
            return True
    return False


def _narrative_similarity(
    query_elements: Dict[str, Tuple[str, ...]],
    candidate_elements: Dict[str, Tuple[str, ...]],
    element_weights: Dict[str, float],
) -> float:
    if not query_elements or not candidate_elements:
        return 0.0

    total_score = 0.0
    total_weight = 0.0

    for element_type, weight in element_weights.items():
        query_items = set(query_elements.get(element_type, []))
        candidate_items = set(candidate_elements.get(element_type, []))

        if query_items and candidate_items:
            overlap = len(query_items & candidate_items)
            union = len(query_items | candidate_items)
            if union > 0:
                jaccard = overlap / union
                total_score += jaccard * weight
                total_weight += weight

    return total_score / max(total_weight, 0.001)


def _freeze_weight_mapping(weights: Dict[str, float]) -> Tuple[Tuple[str, float], ...]:
    return tuple(sorted((str(key), float(value)) for key, value in weights.items()))


@lru_cache(maxsize=2048)
def _score_detailed_cached(
    query_text: str,
    candidate_text: str,
    query_tokens: Tuple[str, ...],
    candidate_tokens: Tuple[str, ...],
    idf_key: Tuple[Tuple[str, float], ...],
    query_concepts: FrozenSet[str],
    candidate_concepts: FrozenSet[str],
    query_elements: Tuple[Tuple[str, Tuple[str, ...]], ...] = (),
    candidate_elements: Tuple[Tuple[str, Tuple[str, ...]], ...] = (),
    narrative_weights_key: Tuple[Tuple[str, float], ...] = (),
    blend_weights_key: Tuple[Tuple[str, float], ...] = (),
) -> float:
    idf = dict(idf_key)
    narrative_weights = dict(narrative_weights_key)
    blend_weights = dict(blend_weights_key)
    
    tfidf  = _tfidf_cosine(query_tokens, candidate_tokens, idf)
    idf_ov = _idf_overlap_ratio(query_tokens, candidate_tokens, idf)

    if query_concepts and candidate_concepts:
        concept_overlap = len(query_concepts & candidate_concepts) / max(len(query_concepts | candidate_concepts), 1)
    else:
        concept_overlap = 0.0

    q_bigrams = _bigrams(query_tokens)
    c_bigrams = _bigrams(candidate_tokens)
    bigram_overlap = _jaccard(q_bigrams, c_bigrams)
    
    token_coef = _overlap_coefficient(set(query_tokens), set(candidate_tokens))

    q_elems_dict = dict(query_elements)
    c_elems_dict = dict(candidate_elements)
    
    narrative_score = _narrative_similarity(q_elems_dict, c_elems_dict, narrative_weights)

    query_norm = _normalize(query_text) if query_text else ""
    candidate_norm = _normalize(candidate_text) if candidate_text else ""
    substring_bonus = 0.0
    if query_norm and candidate_norm and len(query_text) > 10:
        if query_norm in candidate_norm:
            substring_bonus = 0.2

    char_bonus = 0.0
    query_chars = set(q_elems_dict.get("entities", []))
    candidate_chars = set(c_elems_dict.get("entities", []))
    if query_chars and candidate_chars:
        char_overlap = len(query_chars & candidate_chars)
        char_bonus = min(0.3, char_overlap * 0.1)

    query_has_temporal = _contains_any_normalized_term(query_norm, _TEMPORAL_TERMS)
    candidate_has_temporal = _contains_any_normalized_term(candidate_norm, _TEMPORAL_TERMS)
    temporal_bonus = 0.1 if query_has_temporal and candidate_has_temporal else 0.0

    query_has_action = _contains_any_normalized_term(query_norm, _ACTION_TERMS)
    candidate_has_action = _contains_any_normalized_term(candidate_norm, _ACTION_TERMS)
    action_bonus = 0.1 if query_has_action and candidate_has_action else 0.0

    bw = blend_weights
    final_score = (
        bw.get("tfidf",          0.0) * tfidf
        + bw.get("idf_overlap",  0.0) * idf_ov
        + bw.get("concept",      0.0) * concept_overlap
        + bw.get("bigram",       0.0) * bigram_overlap
        + bw.get("token_coef",   0.0) * token_coef
        + bw.get("narrative",    0.0) * narrative_score
        + bw.get("substring",    0.0) * substring_bonus
        + bw.get("entity_bonus", 0.0) * char_bonus
        + bw.get("temporal_bonus", 0.0) * temporal_bonus
        + bw.get("action_bonus",   0.0) * action_bonus
    )

    return min(1.0, final_score)


def _score_detailed_with_weights(
    query_text: str,
    candidate_text: str,
    query_tokens: Tuple[str, ...],
    candidate_tokens: Tuple[str, ...],
    idf_key: Tuple[Tuple[str, float], ...],
    query_concepts: FrozenSet[str],
    candidate_concepts: FrozenSet[str],
    query_elements: Tuple[Tuple[str, Tuple[str, ...]], ...] = (),
    candidate_elements: Tuple[Tuple[str, Tuple[str, ...]], ...] = (),
    narrative_weights: Dict[str, float] | None = None,
    blend_weights: Dict[str, float] | None = None,
) -> float:
    resolved_narrative_weights = _NARRATIVE_WEIGHTS if narrative_weights is None else narrative_weights
    resolved_blend_weights = _BLEND_WEIGHTS if blend_weights is None else blend_weights
    return _score_detailed_cached(
        query_text=query_text,
        candidate_text=candidate_text,
        query_tokens=query_tokens,
        candidate_tokens=candidate_tokens,
        idf_key=idf_key,
        query_concepts=query_concepts,
        candidate_concepts=candidate_concepts,
        query_elements=query_elements,
        candidate_elements=candidate_elements,
        narrative_weights_key=_freeze_weight_mapping(resolved_narrative_weights),
        blend_weights_key=_freeze_weight_mapping(resolved_blend_weights),
    )


def _score_detailed(
    query_text: str,
    candidate_text: str,
    query_tokens: Tuple[str, ...],
    candidate_tokens: Tuple[str, ...],
    idf_key: Tuple[Tuple[str, float], ...],
    query_concepts: FrozenSet[str],
    candidate_concepts: FrozenSet[str],
    query_elements: Tuple[Tuple[str, Tuple[str, ...]], ...] = (),
    candidate_elements: Tuple[Tuple[str, Tuple[str, ...]], ...] = (),
) -> float:
    """
    The primary, computationally-heavier blended scoring function. 
    Combines TF-IDF, concepts, narrative elements, and heuristic bonuses into a final score.
    """
    return _score_detailed_with_weights(
        query_text=query_text,
        candidate_text=candidate_text,
        query_tokens=query_tokens,
        candidate_tokens=candidate_tokens,
        idf_key=idf_key,
        query_concepts=query_concepts,
        candidate_concepts=candidate_concepts,
        query_elements=query_elements,
        candidate_elements=candidate_elements,
    )


def _score_detailed_cache_clear() -> None:
    _score_detailed_cached.cache_clear()


def _score_detailed_cache_info():
    return _score_detailed_cached.cache_info()


_score_detailed.cache_clear = _score_detailed_cache_clear
_score_detailed.cache_info = _score_detailed_cache_info
