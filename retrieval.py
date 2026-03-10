"""
High-level entry points for finding relevant messages in the conversation history.
Coordinates tokenization, concept expansion, fast-filtering, detailed scoring, and clustering.
"""
from __future__ import annotations
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from numbers import Integral
from typing import Dict, FrozenSet, List, Optional, Tuple

from .text_processing import _coerce_text, _tokenize
from .concept_expansion import _expand_concepts
from .narrative_elements import _extract_narrative_elements
from .importance import _calculate_importance_score
from .scoring import _score_fast, _score_detailed, _build_idf
from .concepts import get_word_to_groups

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="acolyterag")
_FAST_CANDIDATE_LIMIT = 30
_MEMORY_TAG = "[RELATED_MEMORY]"


def _detect_query_complexity(tokens: Tuple[str, ...], text: str) -> str:
    n = len(tokens)
    if n <= 4 or len(text) <= 30:
        return "simple"
    if n <= 10 or len(text) <= 80:
        return "moderate"
    return "complex"


def _cluster_similar_memories(
    candidates: List[Dict],
    max_clusters: int = 3,
    precomputed_concepts: Optional[Dict[int, FrozenSet[str]]] = None,
) -> Dict[int, List[int]]:
    if len(candidates) <= max_clusters:
        return {i: [i] for i in range(len(candidates))}

    clusters: Dict[int, List[int]] = {}
    cluster_concepts: Dict[int, FrozenSet[str]] = {}

    for idx in range(len(candidates)):
        concepts = precomputed_concepts.get(idx, frozenset()) if precomputed_concepts else frozenset()
        best_cluster, best_overlap = None, -1.0

        for cid, cconcepts in cluster_concepts.items():
            overlap = (
                len(cconcepts & concepts) / max(len(cconcepts | concepts), 1)
                if cconcepts and concepts else 0.0
            )
            if overlap > best_overlap:
                best_overlap, best_cluster = overlap, cid

        if best_cluster is None or (best_overlap < 0.35 and len(clusters) < max_clusters):
            cid = len(clusters)
            clusters[cid] = [idx]
            cluster_concepts[cid] = concepts
        else:
            clusters[best_cluster].append(idx)
            cluster_concepts[best_cluster] = frozenset(cluster_concepts[best_cluster] | concepts)

    return clusters


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) / 0.75))


def _tag_memory(message: Dict[str, str]) -> Dict[str, str]:
    content = _coerce_text(message.get("content", ""))
    if content.startswith(f"{_MEMORY_TAG} "):
        return {**message, "content": content}
    if content == _MEMORY_TAG:
        return {**message, "content": content}
    if not content:
        return {**message, "content": _MEMORY_TAG}
    return {**message, "content": f"{_MEMORY_TAG} {content}"}


def _is_typo_candidate(token: str) -> bool:
    return len(token) >= 5 and token.isalpha()


def _typo_distance_limit(token: str) -> int:
    return 2 if len(token) >= 8 else 1


@lru_cache(maxsize=8192)
def _osa_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    rows = len(left) + 1
    cols = len(right) + 1
    distances = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        distances[i][0] = i
    for j in range(cols):
        distances[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            substitution_cost = 0 if left[i - 1] == right[j - 1] else 1
            distances[i][j] = min(
                distances[i - 1][j] + 1,
                distances[i][j - 1] + 1,
                distances[i - 1][j - 1] + substitution_cost,
            )
            if (
                i > 1
                and j > 1
                and left[i - 1] == right[j - 2]
                and left[i - 2] == right[j - 1]
            ):
                distances[i][j] = min(
                    distances[i][j],
                    distances[i - 2][j - 2] + 1,
                )

    return distances[-1][-1]


@lru_cache(maxsize=1)
def _get_typo_reference_terms() -> FrozenSet[str]:
    return frozenset(
        term
        for term in get_word_to_groups()
        if _is_typo_candidate(term)
    )


@lru_cache(maxsize=16384)
def _correct_token_typo(token: str, reference_tokens: FrozenSet[str]) -> str:
    if token in reference_tokens or not _is_typo_candidate(token):
        return token

    max_distance = _typo_distance_limit(token)
    best_primary: Optional[Tuple[int, int]] = None
    best_matches: List[str] = []

    for reference in reference_tokens:
        if not _is_typo_candidate(reference) or reference == token:
            continue
        if token[0] != reference[0] or token[:2] != reference[:2]:
            continue
        if abs(len(reference) - len(token)) > max_distance:
            continue

        distance = _osa_distance(token, reference)
        if distance > max_distance:
            continue

        primary = (distance, abs(len(reference) - len(token)))
        if best_primary is None or primary < best_primary:
            best_primary = primary
            best_matches = [reference]
        elif primary == best_primary:
            best_matches.append(reference)

    if best_primary is None or len(best_matches) != 1:
        return token
    return best_matches[0]


def _normalize_tokens_for_typos(
    tokens: Tuple[str, ...],
    reference_tokens: FrozenSet[str],
) -> Tuple[str, ...]:
    if not tokens or not reference_tokens:
        return tokens
    return tuple(_correct_token_typo(token, reference_tokens) for token in tokens)


def _has_semantic_signal(
    query_token_set: FrozenSet[str],
    candidate_token_set: FrozenSet[str],
    query_concepts: FrozenSet[str],
    candidate_concepts: FrozenSet[str],
    query_entities: FrozenSet[str],
    candidate_entities: FrozenSet[str],
) -> bool:
    return bool(
        (query_token_set and candidate_token_set and query_token_set & candidate_token_set)
        or (query_concepts and candidate_concepts and query_concepts & candidate_concepts)
        or (query_entities and candidate_entities and query_entities & candidate_entities)
    )


def _validate_importance_weight(importance_weight: float) -> float:
    try:
        normalized_weight = float(importance_weight)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"importance_weight must be between 0.0 and 1.0, got {importance_weight!r}."
        ) from exc

    if not 0.0 <= normalized_weight <= 1.0:
        raise ValueError(
            f"importance_weight must be between 0.0 and 1.0, got {importance_weight!r}."
        )
    return normalized_weight


def _validate_non_negative_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}.")

    normalized_value = int(value)
    if normalized_value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}.")
    return normalized_value


def _retrieve_related_messages_sync(
    history: List[Dict[str, str]],
    query_text: str,
    max_retrieved: int = 4,
    exclude_last_n: int = 6,
    enable_clustering: bool = True,
    importance_weight: float = 0.3,
    enable_token_based_retrieval: bool = False,
    target_token_count: int = 14000,
    current_token_count: int = 0,
    max_retrieved_for_token_target: int = 50,
) -> List[Dict[str, str]]:
    """
    The synchronous core retrieval algorithm.
    Filters history, applies fast-scoring to get top candidates, ranks them with 
    detailed scoring, and optionally clusters results to maximize topical diversity.
    """
    importance_weight = _validate_importance_weight(importance_weight)
    max_retrieved = _validate_non_negative_integer("max_retrieved", max_retrieved)
    exclude_last_n = _validate_non_negative_integer("exclude_last_n", exclude_last_n)
    target_token_count = _validate_non_negative_integer("target_token_count", target_token_count)
    current_token_count = _validate_non_negative_integer("current_token_count", current_token_count)
    max_retrieved_for_token_target = _validate_non_negative_integer(
        "max_retrieved_for_token_target",
        max_retrieved_for_token_target,
    )
    if not history or not query_text.strip():
        return []

    cutoff = max(0, len(history) - exclude_last_n)
    pool = history[:cutoff]
    if not pool:
        return []

    base_query_tokens = _tokenize(query_text)
    complexity = _detect_query_complexity(base_query_tokens, query_text)
    fast_limit = _FAST_CANDIDATE_LIMIT * 2 if complexity == "complex" else _FAST_CANDIDATE_LIMIT

    
    q_elems_raw = _extract_narrative_elements(query_text)
    query_entities = frozenset(q_elems_raw.get("entities", ()))
    query_elements_tuple = tuple((k, tuple(v)) for k, v in q_elems_raw.items())
    typo_reference_terms = _get_typo_reference_terms()

    n = len(pool)
    q_tokens: List[Tuple[str, ...]] = []
    q_token_sets: List[FrozenSet[str]] = []
    q_concepts: List[FrozenSet[str]] = []
    c_texts: List[str] = []
    c_tokens: List[Tuple[str, ...]] = []
    c_token_sets: List[FrozenSet[str]] = []
    c_importance: List[float] = []
    c_concepts: List[FrozenSet[str]] = []
    c_entities: List[FrozenSet[str]] = []
    c_elements: List[Tuple[Tuple[str, Tuple[str, ...]], ...]] = []

    for msg in pool:
        content = _coerce_text(msg.get("content", ""))
        c_texts.append(content)
        raw_candidate_tokens = _tokenize(content)
        query_reference_tokens = frozenset(raw_candidate_tokens) | typo_reference_terms
        normalized_query_tokens = _normalize_tokens_for_typos(
            base_query_tokens,
            query_reference_tokens,
        )
        candidate_reference_tokens = frozenset(normalized_query_tokens) | typo_reference_terms
        normalized_candidate_tokens = _normalize_tokens_for_typos(
            raw_candidate_tokens,
            candidate_reference_tokens,
        )

        q_tokens.append(normalized_query_tokens)
        q_token_sets.append(frozenset(normalized_query_tokens))
        q_concepts.append(_expand_concepts(normalized_query_tokens))
        c_tokens.append(normalized_candidate_tokens)
        c_token_sets.append(frozenset(normalized_candidate_tokens))
        elements = _extract_narrative_elements(content)
        c_elements.append(tuple((k, tuple(v)) for k, v in elements.items()))
        c_entities.append(frozenset(elements.get("entities", ())))
        c_importance.append(_calculate_importance_score(msg, elements))
        c_concepts.append(_expand_concepts(normalized_candidate_tokens))

    fast_scores = sorted(
        [(_score_fast(q_tokens[i], c_tokens[i], c_importance[i], i / max(n - 1, 1)), i)
         for i in range(n)],
        reverse=True,
    )
    top_fast_indices = [idx for _, idx in fast_scores[:fast_limit]]

    if not top_fast_indices:
        return []

    fast_candidate_tokens = [c_tokens[i] for i in top_fast_indices]
    idf     = _build_idf(fast_candidate_tokens)
    idf_key = tuple(sorted(idf.items()))

    detailed = sorted(
        [
            (
                (1.0 - importance_weight) * _score_detailed(
                    query_text=query_text,
                    candidate_text=c_texts[i],
                    query_tokens=q_tokens[i],
                    candidate_tokens=c_tokens[i],
                    idf_key=idf_key,
                    query_concepts=q_concepts[i],
                    candidate_concepts=c_concepts[i],
                    query_elements=query_elements_tuple,
                    candidate_elements=c_elements[i]
                ) + importance_weight * c_importance[i],
                i,
            )
            for i in top_fast_indices
        ],
        reverse=True,
    )

    detailed = [
        (s, i)
        for s, i in detailed
        if s >= 0.01
        and _has_semantic_signal(
            query_token_set=q_token_sets[i],
            candidate_token_set=c_token_sets[i],
            query_concepts=q_concepts[i],
            candidate_concepts=c_concepts[i],
            query_entities=query_entities,
            candidate_entities=c_entities[i],
        )
    ]
    if not detailed:
        return []

    if enable_token_based_retrieval:
        if max_retrieved <= 0 or max_retrieved_for_token_target <= 0:
            return []
        token_budget = target_token_count - current_token_count
        if token_budget <= 0:
            return []
        selected_indices: List[int] = []
        used = 0
        for _, i in detailed[:max_retrieved_for_token_target]:
            est = _estimate_tokens(c_texts[i])
            if used + est > token_budget:
                continue
            selected_indices.append(i)
            used += est
            if len(selected_indices) >= max_retrieved:
                break
        selected_indices.sort()
        return [_tag_memory(pool[i]) for i in selected_indices]

    if enable_clustering and len(detailed) > max_retrieved:
        concept_map = {k: c_concepts[detailed[k][1]] for k in range(len(detailed))}
        clusters = _cluster_similar_memories(
            [{"_idx": i} for _, i in detailed],
            max_clusters=min(max_retrieved, 4),
            precomputed_concepts=concept_map,
        )
        selected_indices: List[int] = []
        cluster_lists = [
            sorted(members, key=lambda k: detailed[k][0], reverse=True)
            for members in clusters.values()
        ]
        r = 0
        while len(selected_indices) < max_retrieved and any(cluster_lists):
            clist = cluster_lists[r % len(cluster_lists)]
            if clist:
                selected_indices.append(detailed[clist.pop(0)][1])
            r += 1
        selected_indices.sort()
        return [_tag_memory(pool[i]) for i in selected_indices]

    top_indices = sorted(i for _, i in detailed[:max_retrieved])
    return [_tag_memory(pool[i]) for i in top_indices]


def retrieve_related_messages(
    history: List[Dict[str, str]],
    query_text: str,
    max_retrieved: int = 4,
    exclude_last_n: int = 6,
    enable_clustering: bool = True,
    importance_weight: float = 0.3,
    enable_token_based_retrieval: bool = False,
    target_token_count: int = 14000,
    current_token_count: int = 0,
    max_retrieved_for_token_target: int = 50,
) -> List[Dict[str, str]]:
    return _retrieve_related_messages_sync(
        history=history,
        query_text=query_text,
        max_retrieved=max_retrieved,
        exclude_last_n=exclude_last_n,
        enable_clustering=enable_clustering,
        importance_weight=importance_weight,
        enable_token_based_retrieval=enable_token_based_retrieval,
        target_token_count=target_token_count,
        current_token_count=current_token_count,
        max_retrieved_for_token_target=max_retrieved_for_token_target,
    )


async def retrieve_related_messages_async(
    history: List[Dict[str, str]],
    query_text: str,
    max_retrieved: int = 4,
    exclude_last_n: int = 6,
    enable_clustering: bool = True,
    importance_weight: float = 0.3,
    enable_token_based_retrieval: bool = False,
    target_token_count: int = 14000,
    current_token_count: int = 0,
    max_retrieved_for_token_target: int = 50,
) -> List[Dict[str, str]]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor,
        lambda: _retrieve_related_messages_sync(
            history=history,
            query_text=query_text,
            max_retrieved=max_retrieved,
            exclude_last_n=exclude_last_n,
            enable_clustering=enable_clustering,
            importance_weight=importance_weight,
            enable_token_based_retrieval=enable_token_based_retrieval,
            target_token_count=target_token_count,
            current_token_count=current_token_count,
            max_retrieved_for_token_target=max_retrieved_for_token_target,
        ),
    )
