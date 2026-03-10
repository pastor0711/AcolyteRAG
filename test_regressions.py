import ast
import http.client
import json
from pathlib import Path
import shutil
import sys
import threading
from http.server import HTTPServer

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent
_PACKAGE_ROOT = str(_PROJECT_ROOT.parent)
_TEST_TEMP_ROOT = _PROJECT_ROOT / ".venv" / "tmp-tests"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)

import concept_manager
from acolyterag.analysis import create_memory_index, get_memory_statistics, summarize_conversation_window
import acolyterag.concept_expansion as concept_expansion
import acolyterag.concepts as concepts
import acolyterag.narrative_elements as narrative_elements
import acolyterag.retrieval as retrieval
import acolyterag.scoring as scoring
from acolyterag.text_processing import _tokenize


def _score_with_loaded_modules(query_text: str, candidate_text: str) -> float:
    q_tokens = _tokenize(query_text)
    c_tokens = _tokenize(candidate_text)
    idf = scoring._build_idf([q_tokens, c_tokens])
    idf_key = tuple(sorted(idf.items()))
    q_concepts = concept_expansion._expand_concepts(q_tokens)
    c_concepts = concept_expansion._expand_concepts(c_tokens)
    q_elements = narrative_elements._extract_narrative_elements(query_text)
    c_elements = narrative_elements._extract_narrative_elements(candidate_text)
    return scoring._score_detailed(
        query_text=query_text,
        candidate_text=candidate_text,
        query_tokens=q_tokens,
        candidate_tokens=c_tokens,
        idf_key=idf_key,
        query_concepts=q_concepts,
        candidate_concepts=c_concepts,
        query_elements=tuple((k, tuple(v)) for k, v in q_elements.items()),
        candidate_elements=tuple((k, tuple(v)) for k, v in c_elements.items()),
    )


def _score_with_loaded_modules_and_weights(
    query_text: str,
    candidate_text: str,
    narrative_weights: dict,
    blend_weights: dict,
) -> float:
    q_tokens = _tokenize(query_text)
    c_tokens = _tokenize(candidate_text)
    idf = scoring._build_idf([q_tokens, c_tokens])
    idf_key = tuple(sorted(idf.items()))
    q_concepts = concept_expansion._expand_concepts(q_tokens)
    c_concepts = concept_expansion._expand_concepts(c_tokens)
    q_elements = narrative_elements._extract_narrative_elements(query_text)
    c_elements = narrative_elements._extract_narrative_elements(candidate_text)
    return scoring._score_detailed_with_weights(
        query_text=query_text,
        candidate_text=candidate_text,
        query_tokens=q_tokens,
        candidate_tokens=c_tokens,
        idf_key=idf_key,
        query_concepts=q_concepts,
        candidate_concepts=c_concepts,
        query_elements=tuple((k, tuple(v)) for k, v in q_elements.items()),
        candidate_elements=tuple((k, tuple(v)) for k, v in c_elements.items()),
        narrative_weights=narrative_weights,
        blend_weights=blend_weights,
    )


def _score_with_isolated_bonus(query_text: str, candidate_text: str, bonus_key: str) -> float:
    original_blend_weights = dict(scoring._BLEND_WEIGHTS)
    try:
        for key in scoring._BLEND_WEIGHTS:
            scoring._BLEND_WEIGHTS[key] = 0.0
        scoring._BLEND_WEIGHTS[bonus_key] = 1.0
        scoring._score_detailed.cache_clear()
        return _score_with_loaded_modules(query_text, candidate_text)
    finally:
        scoring._BLEND_WEIGHTS.clear()
        scoring._BLEND_WEIGHTS.update(original_blend_weights)
        scoring._score_detailed.cache_clear()


def _prepare_temp_dir(name: str) -> Path:
    _TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    temp_dir = _TEST_TEMP_ROOT / name
    shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def _snapshot_runtime_concepts() -> dict[str, set[str]]:
    return {
        group: set(words)
        for group, words in concepts._registry.items()
    }


def _restore_runtime_concepts(snapshot: dict[str, set[str]]) -> None:
    concepts._registry.clear()
    concepts._registry.update({
        group: set(words)
        for group, words in snapshot.items()
    })
    concepts._clear_runtime_concept_caches()


def _snapshot_runtime_scoring() -> tuple[dict[str, float], dict[str, float], dict[str, set[str]]]:
    return (
        dict(scoring._NARRATIVE_WEIGHTS),
        dict(scoring._BLEND_WEIGHTS),
        {
            dim: set(values)
            for dim, values in narrative_elements._SCORING_GROUPS.items()
        },
    )


def _restore_runtime_scoring(
    snapshot: tuple[dict[str, float], dict[str, float], dict[str, set[str]]]
) -> None:
    narrative_weights, blend_weights, groups = snapshot
    modules = concept_manager._load_runtime_modules(reload_modules=False)
    concept_manager._apply_runtime_scoring_state(
        modules,
        narrative_weights,
        blend_weights,
        groups,
    )


def test_summarize_conversation_window_respects_explicit_zero_end():
    history = [
        {"role": "user", "content": "This should not appear in an empty summary."},
    ]

    assert summarize_conversation_window(history, window_start=0, window_end=0) == ""


def test_preview_score_reloads_file_backed_scoring_weights():
    query_text = "python web framework comparison"
    candidate_text = "Python web frameworks are useful for backend apps."
    original_blend_weights = dict(scoring._BLEND_WEIGHTS)
    baseline_score = concept_manager._preview_score(query_text, candidate_text)

    try:
        for key in scoring._BLEND_WEIGHTS:
            scoring._BLEND_WEIGHTS[key] = 0.0

        reloaded_score = concept_manager._preview_score(query_text, candidate_text)

        assert scoring._BLEND_WEIGHTS == original_blend_weights
        assert reloaded_score == baseline_score
    finally:
        concept_manager._preview_score(query_text, candidate_text)


def test_preview_score_reloads_text_processing_module(monkeypatch):
    reloaded_modules = []
    original_reload = concept_manager.importlib.reload

    def _tracking_reload(module):
        reloaded_modules.append(module.__name__)
        return original_reload(module)

    monkeypatch.setattr(concept_manager.importlib, "reload", _tracking_reload)

    concept_manager._preview_score("Python framework", "Python framework")

    assert "acolyterag.text_processing" in reloaded_modules


def test_concept_manager_returns_400_for_invalid_json():
    server = HTTPServer(("127.0.0.1", 0), concept_manager.Handler)
    worker = threading.Thread(target=server.handle_request)
    worker.start()

    conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
    try:
        conn.request(
            "POST",
            "/api/add_group",
            body="not json",
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
    finally:
        conn.close()
        server.server_close()
        worker.join(timeout=5)

    assert response.status == 400
    assert payload == {"error": "Invalid JSON payload."}


def test_coerce_scoring_weights_rejects_non_finite_values():
    for invalid_value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="finite number"):
            concept_manager._coerce_scoring_weights(
                {"_BLEND_WEIGHTS": {"tfidf": invalid_value}}
            )


def test_normalize_concept_words_rejects_scalar_string_payload():
    with pytest.raises(ValueError, match="array of strings"):
        concept_manager._normalize_concept_words("python")


def test_concept_manager_returns_400_for_non_finite_json_number():
    server = HTTPServer(("127.0.0.1", 0), concept_manager.Handler)
    worker = threading.Thread(target=server.handle_request)
    worker.start()

    conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
    try:
        conn.request(
            "POST",
            "/api/save_scoring_bundle",
            body='{"scoring":{"_BLEND_WEIGHTS":{"tfidf":NaN}},"groups":{}}',
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
    finally:
        conn.close()
        server.server_close()
        worker.join(timeout=5)

    assert response.status == 400
    assert payload == {"error": "Invalid numeric constant: NaN."}


def test_concept_manager_returns_400_for_missing_required_add_words_field():
    server = HTTPServer(("127.0.0.1", 0), concept_manager.Handler)
    worker = threading.Thread(target=server.handle_request)
    worker.start()

    conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
    try:
        conn.request(
            "POST",
            "/api/add_words",
            body=json.dumps({"group": "demo"}),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
    finally:
        conn.close()
        server.server_close()
        worker.join(timeout=5)

    assert response.status == 400
    assert payload == {"error": "Missing required field: words."}


def test_concept_manager_returns_400_for_scalar_words_payload():
    server = HTTPServer(("127.0.0.1", 0), concept_manager.Handler)
    worker = threading.Thread(target=server.handle_request)
    worker.start()

    conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
    try:
        conn.request(
            "POST",
            "/api/add_words",
            body=json.dumps({"group": "demo", "words": "python"}),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
    finally:
        conn.close()
        server.server_close()
        worker.join(timeout=5)

    assert response.status == 400
    assert payload == {"error": "words must be an array of strings."}


def test_concept_manager_returns_400_for_add_words_on_unknown_group(monkeypatch):
    temp_dir = _prepare_temp_dir("add_words_unknown_group")
    concepts_path = temp_dir / "concepts.py"
    concepts_path.write_text(
        "from typing import Dict, List\n\n"
        "_CORE_CONCEPTS: Dict[str, List[str]] = {\n"
        "    'conflict': ['fight'],\n"
        "}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(concept_manager, "CONCEPTS", concepts_path)

    server = HTTPServer(("127.0.0.1", 0), concept_manager.Handler)
    worker = threading.Thread(target=server.handle_request)
    worker.start()

    conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
    try:
        conn.request(
            "POST",
            "/api/add_words",
            body=json.dumps({"group": "missing", "words": ["fight"]}),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
    finally:
        conn.close()
        server.server_close()
        worker.join(timeout=5)

    assert response.status == 400
    assert payload == {"error": "Unknown concept group: missing."}
    assert concept_manager._load_concepts() == {"conflict": ["fight"]}


def test_concept_manager_returns_400_for_remove_word_on_unknown_group(monkeypatch):
    temp_dir = _prepare_temp_dir("remove_word_unknown_group")
    concepts_path = temp_dir / "concepts.py"
    concepts_path.write_text(
        "from typing import Dict, List\n\n"
        "_CORE_CONCEPTS: Dict[str, List[str]] = {\n"
        "    'conflict': ['fight'],\n"
        "}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(concept_manager, "CONCEPTS", concepts_path)

    server = HTTPServer(("127.0.0.1", 0), concept_manager.Handler)
    worker = threading.Thread(target=server.handle_request)
    worker.start()

    conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
    try:
        conn.request(
            "POST",
            "/api/remove_word",
            body=json.dumps({"group": "missing", "word": "fight"}),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
    finally:
        conn.close()
        server.server_close()
        worker.join(timeout=5)

    assert response.status == 400
    assert payload == {"error": "Unknown concept group: missing."}
    assert concept_manager._load_concepts() == {"conflict": ["fight"]}


def test_concept_manager_returns_400_for_duplicate_add_group(monkeypatch):
    temp_dir = _prepare_temp_dir("duplicate_add_group")
    concepts_path = temp_dir / "concepts.py"
    concepts_path.write_text(
        "from typing import Dict, List\n\n"
        "_CORE_CONCEPTS: Dict[str, List[str]] = {\n"
        "    'conflict': ['fight'],\n"
        "}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(concept_manager, "CONCEPTS", concepts_path)

    server = HTTPServer(("127.0.0.1", 0), concept_manager.Handler)
    worker = threading.Thread(target=server.handle_request)
    worker.start()

    conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
    try:
        conn.request(
            "POST",
            "/api/add_group",
            body=json.dumps({"group": "conflict", "words": ["war"]}),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
    finally:
        conn.close()
        server.server_close()
        worker.join(timeout=5)

    assert response.status == 400
    assert payload == {"error": "Concept group already exists: conflict."}
    assert concept_manager._load_concepts() == {"conflict": ["fight"]}


def test_concept_manager_returns_400_for_non_object_scoring_section():
    server = HTTPServer(("127.0.0.1", 0), concept_manager.Handler)
    worker = threading.Thread(target=server.handle_request)
    worker.start()

    conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
    try:
        conn.request(
            "POST",
            "/api/save_scoring_bundle",
            body=json.dumps({"scoring": [], "groups": {}}),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
    finally:
        conn.close()
        server.server_close()
        worker.join(timeout=5)

    assert response.status == 400
    assert payload == {"error": "scoring must be an object."}


def test_temporal_bonus_requires_exact_normalized_term_match():
    exact_score = _score_with_isolated_bonus(
        "Should we revisit this tomorrow?",
        "We will revisit this tomorrow morning.",
        "temporal_bonus",
    )
    false_positive_score = _score_with_isolated_bonus(
        "unrelated query",
        "We will revisit this tomorrow morning.",
        "temporal_bonus",
    )

    assert exact_score == 0.1
    assert false_positive_score == 0.0


def test_action_bonus_requires_query_and_candidate_alignment():
    exact_score = _score_with_isolated_bonus(
        "What took place during the ceremony?",
        "The ceremony took place at sunrise.",
        "action_bonus",
    )
    false_positive_score = _score_with_isolated_bonus(
        "unrelated query",
        "The ceremony took place at sunrise.",
        "action_bonus",
    )

    assert exact_score == 0.1
    assert false_positive_score == 0.0


def test_zero_match_query_filters_temporal_and_action_only_candidates():
    history = [
        {"role": "assistant", "content": "We will meet tomorrow at dawn."},
        {"role": "assistant", "content": "The ceremony took place at sunrise."},
        {"role": "user", "content": "recent"},
    ]

    results = retrieval.retrieve_related_messages(
        history,
        query_text="quantum teleportation",
        exclude_last_n=1,
        importance_weight=0.0,
    )

    assert results == []


def test_expand_concepts_does_not_prefix_match_unknown_long_token():
    assert concept_expansion._expand_concepts(_tokenize("companionate")) == frozenset()


def test_retrieval_corrects_query_typo_against_candidate_tokens():
    history = [
        {"role": "assistant", "content": "deployment checklist"},
        {"role": "user", "content": "recent"},
    ]

    results = retrieval.retrieve_related_messages(
        history,
        query_text="deploymnt checklist",
        exclude_last_n=1,
        enable_clustering=False,
        importance_weight=0.0,
    )

    assert results == [
        {
            "role": "assistant",
            "content": "[RELATED_MEMORY] deployment checklist",
        }
    ]


def test_retrieval_corrects_candidate_typo_against_query_tokens():
    history = [
        {"role": "assistant", "content": "deploymnt checklist"},
        {"role": "user", "content": "recent"},
    ]

    results = retrieval.retrieve_related_messages(
        history,
        query_text="deployment checklist",
        exclude_last_n=1,
        enable_clustering=False,
        importance_weight=0.0,
    )

    assert results == [
        {
            "role": "assistant",
            "content": "[RELATED_MEMORY] deploymnt checklist",
        }
    ]


def test_retrieval_does_not_double_tag_existing_related_memory_prefix():
    history = [
        {"role": "assistant", "content": "[RELATED_MEMORY] alpha beta"},
        {"role": "user", "content": "recent"},
    ]

    results = retrieval.retrieve_related_messages(
        history,
        query_text="alpha beta",
        exclude_last_n=1,
        enable_clustering=False,
        importance_weight=0.0,
    )

    assert results == [
        {
            "role": "assistant",
            "content": "[RELATED_MEMORY] alpha beta",
        }
    ]


def test_register_concepts_invalidates_typo_reference_cache_after_preview_reload():
    history = [
        {"role": "assistant", "content": "orchestration patterns for agents"},
        {"role": "user", "content": "recent"},
    ]

    retrieval._get_typo_reference_terms.cache_clear()
    retrieval._correct_token_typo.cache_clear()
    concepts.unregister_concepts("frameworkx")
    try:
        concept_manager._preview_score("Python framework", "Python framework")

        before = retrieval.retrieve_related_messages(
            history,
            query_text="langgrph",
            exclude_last_n=1,
            enable_clustering=False,
            importance_weight=0.0,
        )

        concepts.register_concepts("frameworkx", ["langgraph", "orchestration"])

        after = retrieval.retrieve_related_messages(
            history,
            query_text="langgrph",
            exclude_last_n=1,
            enable_clustering=False,
            importance_weight=0.0,
        )

        assert before == []
        assert after == [
            {
                "role": "assistant",
                "content": "[RELATED_MEMORY] orchestration patterns for agents",
            }
        ]
    finally:
        concepts.unregister_concepts("frameworkx")
        retrieval._get_typo_reference_terms.cache_clear()
        retrieval._correct_token_typo.cache_clear()


def test_retrieval_typo_correction_avoids_old_prefix_false_positive():
    history = [
        {"role": "assistant", "content": "The company office expanded downtown."},
        {"role": "assistant", "content": "Her companion showed deep loyalty."},
        {"role": "user", "content": "recent"},
    ]

    results = retrieval.retrieve_related_messages(
        history,
        query_text="compaion loyalty",
        max_retrieved=1,
        exclude_last_n=1,
        enable_clustering=False,
        importance_weight=0.0,
    )

    assert results == [
        {
            "role": "assistant",
            "content": "[RELATED_MEMORY] Her companion showed deep loyalty.",
        }
    ]


def test_entity_bonus_ignores_sentence_start_imperative_false_positive():
    false_positive_score = _score_with_isolated_bonus(
        "Tell me about Python.",
        "Tell me about Django.",
        "entity_bonus",
    )
    real_entity_score = _score_with_isolated_bonus(
        "Alice asked about Python.",
        "Alice reviewed Django.",
        "entity_bonus",
    )

    assert false_positive_score == 0.0
    assert real_entity_score == 0.1


def test_cluster_similar_memories_updates_cluster_representatives():
    clusters = retrieval._cluster_similar_memories(
        [{}, {}, {}, {}],
        max_clusters=2,
        precomputed_concepts={
            0: frozenset({"alpha"}),
            1: frozenset({"alpha", "beta"}),
            2: frozenset({"beta"}),
            3: frozenset({"gamma"}),
        },
    )

    assert sorted(sorted(members) for members in clusters.values()) == [[0, 1, 2], [3]]


def test_save_scoring_groups_lowercases_mixed_case_concepts(monkeypatch):
    temp_dir = _prepare_temp_dir("save_scoring_groups")
    narrative_path = temp_dir / "narrative_elements.py"
    narrative_path.write_text(
        "from typing import Dict, Set\n\n"
        "_SCORING_GROUPS: Dict[str, Set[str]] = {\n"
        "    'emotions': {'anger'},\n"
        "}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(concept_manager, "NARRATIVE", narrative_path)

    concept_manager._save_scoring_groups({"emotions": ["Anger", "FEAR", "anger"]})

    assert concept_manager._load_scoring_groups()["emotions"] == ["anger", "fear"]
    assert "Anger" not in narrative_path.read_text(encoding="utf-8")


def test_save_scoring_bundle_rolls_back_when_second_write_fails(monkeypatch):
    temp_dir = _prepare_temp_dir("save_scoring_bundle")
    scoring_path = temp_dir / "scoring.py"
    narrative_path = temp_dir / "narrative_elements.py"
    scoring_source = (
        "from typing import Dict\n\n"
        "_NARRATIVE_WEIGHTS: Dict[str, float] = {\n"
        "    'emotions': 0.3,\n"
        "}\n\n"
        "_BLEND_WEIGHTS: Dict[str, float] = {\n"
        "    'tfidf': 0.2,\n"
        "}\n"
    )
    narrative_source = (
        "from typing import Dict, Set\n\n"
        "_SCORING_GROUPS: Dict[str, Set[str]] = {\n"
        "    'emotions': {'anger'},\n"
        "}\n"
    )
    scoring_path.write_text(scoring_source, encoding="utf-8")
    narrative_path.write_text(narrative_source, encoding="utf-8")
    monkeypatch.setattr(concept_manager, "SCORING", scoring_path)
    monkeypatch.setattr(concept_manager, "NARRATIVE", narrative_path)

    original_write_text_file = concept_manager._write_text_file
    narrative_failure_state = {"remaining": 1}

    def _fail_on_narrative_write(path, text):
        if path == concept_manager.NARRATIVE and narrative_failure_state["remaining"] > 0:
            narrative_failure_state["remaining"] -= 1
            raise OSError("simulated narrative write failure")
        original_write_text_file(path, text)

    monkeypatch.setattr(concept_manager, "_write_text_file", _fail_on_narrative_write)

    with pytest.raises(OSError, match="simulated narrative write failure"):
        concept_manager._save_scoring_bundle(
            {
                "_NARRATIVE_WEIGHTS": {"emotions": 0.5},
                "_BLEND_WEIGHTS": {"tfidf": 0.9},
            },
            {"emotions": ["Anger", "Fear"]},
        )

    assert scoring_path.read_text(encoding="utf-8") == scoring_source
    assert narrative_path.read_text(encoding="utf-8") == narrative_source


def test_save_scoring_weights_rolls_back_when_runtime_refresh_fails(monkeypatch):
    temp_dir = _prepare_temp_dir("save_scoring_weights_rollback")
    scoring_path = temp_dir / "scoring.py"
    narrative_path = temp_dir / "narrative_elements.py"
    scoring_source = (
        "from typing import Dict\n\n"
        "_NARRATIVE_WEIGHTS: Dict[str, float] = {\n"
        "    'emotions': 0.3,\n"
        "}\n\n"
        "_BLEND_WEIGHTS: Dict[str, float] = {\n"
        "    'tfidf': 0.2,\n"
        "    'narrative': 0.25,\n"
        "}\n"
    )
    narrative_source = (
        "from typing import Dict, Set\n\n"
        "_SCORING_GROUPS: Dict[str, Set[str]] = {\n"
        "    'emotions': {'anger'},\n"
        "}\n"
    )
    scoring_path.write_text(scoring_source, encoding="utf-8")
    narrative_path.write_text(narrative_source, encoding="utf-8")
    monkeypatch.setattr(concept_manager, "SCORING", scoring_path)
    monkeypatch.setattr(concept_manager, "NARRATIVE", narrative_path)

    def _fail_refresh(*_args, **_kwargs):
        raise RuntimeError("simulated scoring refresh failure")

    monkeypatch.setattr(concept_manager, "_refresh_runtime_scoring_state", _fail_refresh)

    with pytest.raises(RuntimeError, match="simulated scoring refresh failure"):
        concept_manager._save_scoring_weights(
            {"emotions": 0.5},
            {"tfidf": 0.9, "narrative": 0.1},
        )

    assert scoring_path.read_text(encoding="utf-8") == scoring_source


def test_save_scoring_groups_rolls_back_when_runtime_refresh_fails(monkeypatch):
    temp_dir = _prepare_temp_dir("save_scoring_groups_rollback")
    scoring_path = temp_dir / "scoring.py"
    narrative_path = temp_dir / "narrative_elements.py"
    scoring_source = (
        "from typing import Dict\n\n"
        "_NARRATIVE_WEIGHTS: Dict[str, float] = {\n"
        "    'emotions': 0.3,\n"
        "}\n\n"
        "_BLEND_WEIGHTS: Dict[str, float] = {\n"
        "    'narrative': 0.25,\n"
        "}\n"
    )
    narrative_source = (
        "from typing import Dict, Set\n\n"
        "_SCORING_GROUPS: Dict[str, Set[str]] = {\n"
        "    'emotions': {'anger'},\n"
        "}\n"
    )
    scoring_path.write_text(scoring_source, encoding="utf-8")
    narrative_path.write_text(narrative_source, encoding="utf-8")
    monkeypatch.setattr(concept_manager, "SCORING", scoring_path)
    monkeypatch.setattr(concept_manager, "NARRATIVE", narrative_path)

    def _fail_refresh(*_args, **_kwargs):
        raise RuntimeError("simulated scoring-group refresh failure")

    monkeypatch.setattr(concept_manager, "_refresh_runtime_scoring_state", _fail_refresh)

    with pytest.raises(RuntimeError, match="simulated scoring-group refresh failure"):
        concept_manager._save_scoring_groups({"emotions": ["anger", "fear"]})

    assert narrative_path.read_text(encoding="utf-8") == narrative_source


def test_save_scoring_bundle_rolls_back_when_runtime_refresh_fails(monkeypatch):
    temp_dir = _prepare_temp_dir("save_scoring_bundle_rollback")
    scoring_path = temp_dir / "scoring.py"
    narrative_path = temp_dir / "narrative_elements.py"
    scoring_source = (
        "from typing import Dict\n\n"
        "_NARRATIVE_WEIGHTS: Dict[str, float] = {\n"
        "    'actions': 0.3,\n"
        "}\n\n"
        "_BLEND_WEIGHTS: Dict[str, float] = {\n"
        "    'tfidf': 0.2,\n"
        "    'narrative': 0.25,\n"
        "}\n"
    )
    narrative_source = (
        "from typing import Dict, Set\n\n"
        "_SCORING_GROUPS: Dict[str, Set[str]] = {\n"
        "    'actions': {'meeting'},\n"
        "}\n"
    )
    scoring_path.write_text(scoring_source, encoding="utf-8")
    narrative_path.write_text(narrative_source, encoding="utf-8")
    monkeypatch.setattr(concept_manager, "SCORING", scoring_path)
    monkeypatch.setattr(concept_manager, "NARRATIVE", narrative_path)

    original_runtime_scoring = _snapshot_runtime_scoring()
    expected_runtime_scoring = (
        {"actions": 0.3},
        {"tfidf": 0.2, "narrative": 0.25},
        {"actions": {"meeting"}},
    )

    def _fail_refresh(*_args, **_kwargs):
        raise RuntimeError("simulated scoring bundle refresh failure")

    monkeypatch.setattr(concept_manager, "_refresh_runtime_scoring_state", _fail_refresh)

    try:
        with pytest.raises(RuntimeError, match="simulated scoring bundle refresh failure"):
            concept_manager._save_scoring_bundle(
                {
                    "_NARRATIVE_WEIGHTS": {"actions": 0.7},
                    "_BLEND_WEIGHTS": {"tfidf": 0.9},
                },
                {"actions": ["authority"]},
            )

        assert scoring_path.read_text(encoding="utf-8") == scoring_source
        assert narrative_path.read_text(encoding="utf-8") == narrative_source
        assert _snapshot_runtime_scoring() == expected_runtime_scoring
    finally:
        _restore_runtime_scoring(original_runtime_scoring)


def test_delete_concept_group_rolls_back_when_scoring_group_write_fails(monkeypatch):
    temp_dir = _prepare_temp_dir("delete_concept_group_rollback")
    concepts_path = temp_dir / "concepts.py"
    scoring_path = temp_dir / "scoring.py"
    narrative_path = temp_dir / "narrative_elements.py"
    concepts_source = (
        "from typing import Dict, List\n\n"
        "_CORE_CONCEPTS: Dict[str, List[str]] = {\n"
        "    'conflict': ['fight'],\n"
        "    'travel': ['journey'],\n"
        "}\n"
    )
    scoring_source = (
        "from typing import Dict\n\n"
        "_NARRATIVE_WEIGHTS: Dict[str, float] = {\n"
        "    'actions': 0.3,\n"
        "}\n\n"
        "_BLEND_WEIGHTS: Dict[str, float] = {\n"
        "    'narrative': 0.25,\n"
        "}\n"
    )
    narrative_source = (
        "from typing import Dict, Set\n\n"
        "_SCORING_GROUPS: Dict[str, Set[str]] = {\n"
        "    'actions': {'conflict', 'travel'},\n"
        "}\n"
    )
    concepts_path.write_text(concepts_source, encoding="utf-8")
    scoring_path.write_text(scoring_source, encoding="utf-8")
    narrative_path.write_text(narrative_source, encoding="utf-8")
    monkeypatch.setattr(concept_manager, "CONCEPTS", concepts_path)
    monkeypatch.setattr(concept_manager, "SCORING", scoring_path)
    monkeypatch.setattr(concept_manager, "NARRATIVE", narrative_path)

    original_write_text_file = concept_manager._write_text_file
    narrative_failure_state = {"remaining": 1}

    def _fail_on_narrative_write(path, text):
        if path == concept_manager.NARRATIVE and narrative_failure_state["remaining"] > 0:
            narrative_failure_state["remaining"] -= 1
            raise OSError("simulated delete narrative write failure")
        original_write_text_file(path, text)

    monkeypatch.setattr(concept_manager, "_write_text_file", _fail_on_narrative_write)

    with pytest.raises(OSError, match="simulated delete narrative write failure"):
        concept_manager._delete_concept_group("conflict")

    assert concepts_path.read_text(encoding="utf-8") == concepts_source
    assert narrative_path.read_text(encoding="utf-8") == narrative_source


def test_delete_concept_group_rolls_back_when_runtime_refresh_fails(monkeypatch):
    temp_dir = _prepare_temp_dir("delete_concept_group_refresh_rollback")
    concepts_path = temp_dir / "concepts.py"
    scoring_path = temp_dir / "scoring.py"
    narrative_path = temp_dir / "narrative_elements.py"
    concepts_source = (
        "from typing import Dict, List\n\n"
        "_CORE_CONCEPTS: Dict[str, List[str]] = {\n"
        "    'conflict': ['fight'],\n"
        "    'travel': ['journey'],\n"
        "}\n"
    )
    scoring_source = (
        "from typing import Dict\n\n"
        "_NARRATIVE_WEIGHTS: Dict[str, float] = {\n"
        "    'actions': 0.3,\n"
        "}\n\n"
        "_BLEND_WEIGHTS: Dict[str, float] = {\n"
        "    'narrative': 0.25,\n"
        "}\n"
    )
    narrative_source = (
        "from typing import Dict, Set\n\n"
        "_SCORING_GROUPS: Dict[str, Set[str]] = {\n"
        "    'actions': {'conflict', 'travel'},\n"
        "}\n"
    )
    concepts_path.write_text(concepts_source, encoding="utf-8")
    scoring_path.write_text(scoring_source, encoding="utf-8")
    narrative_path.write_text(narrative_source, encoding="utf-8")
    monkeypatch.setattr(concept_manager, "CONCEPTS", concepts_path)
    monkeypatch.setattr(concept_manager, "SCORING", scoring_path)
    monkeypatch.setattr(concept_manager, "NARRATIVE", narrative_path)

    original_runtime_concepts = _snapshot_runtime_concepts()
    original_runtime_scoring = _snapshot_runtime_scoring()
    expected_runtime_concepts = {
        "conflict": {"fight"},
        "travel": {"journey"},
    }
    expected_runtime_scoring = (
        {"actions": 0.3},
        {"narrative": 0.25},
        {"actions": {"conflict", "travel"}},
    )

    def _fail_refresh(*_args, **_kwargs):
        raise RuntimeError("simulated delete scoring refresh failure")

    monkeypatch.setattr(concept_manager, "_refresh_runtime_scoring_state", _fail_refresh)

    try:
        with pytest.raises(RuntimeError, match="simulated delete scoring refresh failure"):
            concept_manager._delete_concept_group("conflict")

        assert concepts_path.read_text(encoding="utf-8") == concepts_source
        assert narrative_path.read_text(encoding="utf-8") == narrative_source
        assert _snapshot_runtime_concepts() == expected_runtime_concepts
        assert _snapshot_runtime_scoring() == expected_runtime_scoring
    finally:
        _restore_runtime_concepts(original_runtime_concepts)
        _restore_runtime_scoring(original_runtime_scoring)


def test_coerce_scoring_weights_preserves_missing_sections_and_values(monkeypatch):
    temp_dir = _prepare_temp_dir("coerce_scoring_weights")
    scoring_path = temp_dir / "scoring.py"
    scoring_path.write_text(
        "from typing import Dict\n\n"
        "_NARRATIVE_WEIGHTS: Dict[str, float] = {\n"
        "    'emotions': 0.3,\n"
        "}\n\n"
        "_BLEND_WEIGHTS: Dict[str, float] = {\n"
        "    'tfidf': 0.2,\n"
        "    'narrative': 0.25,\n"
        "}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(concept_manager, "SCORING", scoring_path)

    narrative, blend = concept_manager._coerce_scoring_weights(
        {"_BLEND_WEIGHTS": {"tfidf": 0.9}}
    )
    concept_manager._save_scoring_weights(narrative, blend)

    saved = scoring_path.read_text(encoding="utf-8")
    ast.parse(saved)

    assert concept_manager._load_scoring_weights() == {
        "_NARRATIVE_WEIGHTS": {"emotions": 0.3},
        "_BLEND_WEIGHTS": {"tfidf": 0.9, "narrative": 0.25},
    }


def test_render_scoring_weights_source_serializes_empty_dicts_as_valid_python():
    src = (
        "from typing import Dict\n\n"
        "_NARRATIVE_WEIGHTS: Dict[str, float] = {\n"
        "    'emotions': 0.3,\n"
        "}\n\n"
        "_BLEND_WEIGHTS: Dict[str, float] = {\n"
        "    'tfidf': 0.2,\n"
        "}\n"
    )

    updated = concept_manager._render_scoring_weights_source(src, {}, {})

    ast.parse(updated)
    assert "_NARRATIVE_WEIGHTS: Dict[str, float] = {}" in updated
    assert "_BLEND_WEIGHTS: Dict[str, float] = {}" in updated


def test_load_and_save_concepts_normalize_mixed_case_groups(monkeypatch):
    temp_dir = _prepare_temp_dir("normalize_concepts")
    concepts_path = temp_dir / "concepts.py"
    concepts_path.write_text(
        "from typing import Dict, List\n\n"
        "_CORE_CONCEPTS: Dict[str, List[str]] = {\n"
        "    'MixedGroup': ['Alpha'],\n"
        "    'mixedgroup': ['beta'],\n"
        "    'Other': ['Gamma'],\n"
        "}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(concept_manager, "CONCEPTS", concepts_path)
    original_registry = _snapshot_runtime_concepts()

    try:
        loaded = concept_manager._load_concepts()

        assert loaded == {
            "mixedgroup": ["alpha", "beta"],
            "other": ["gamma"],
        }

        concept_manager._save_concepts({" MixedGroup ": ["Alpha", "BETA"], "OTHER": ["Gamma"]})

        saved = concepts_path.read_text(encoding="utf-8")
        assert "'mixedgroup'" in saved
        assert "'other'" in saved
        assert "'MixedGroup'" not in saved
        assert "'OTHER'" not in saved
    finally:
        _restore_runtime_concepts(original_registry)


def test_save_concepts_rolls_back_when_runtime_refresh_fails(monkeypatch):
    temp_dir = _prepare_temp_dir("save_concepts_rollback")
    concepts_path = temp_dir / "concepts.py"
    concepts_source = (
        "from typing import Dict, List\n\n"
        "_CORE_CONCEPTS: Dict[str, List[str]] = {\n"
        "    'conflict': ['fight'],\n"
        "    'travel': ['journey'],\n"
        "}\n"
    )
    concepts_path.write_text(concepts_source, encoding="utf-8")
    monkeypatch.setattr(concept_manager, "CONCEPTS", concepts_path)

    def _fail_refresh(*_args, **_kwargs):
        raise RuntimeError("simulated concept refresh failure")

    monkeypatch.setattr(concept_manager, "_refresh_runtime_concept_state", _fail_refresh)

    with pytest.raises(RuntimeError, match="simulated concept refresh failure"):
        concept_manager._save_concepts({"conflict": ["fight"], "travel": ["voyage"]})

    assert concepts_path.read_text(encoding="utf-8") == concepts_source


def test_delete_concept_group_removes_scoring_references(monkeypatch):
    temp_dir = _prepare_temp_dir("delete_concept_group_cleanup")
    concepts_path = temp_dir / "concepts.py"
    scoring_path = temp_dir / "scoring.py"
    narrative_path = temp_dir / "narrative_elements.py"
    concepts_path.write_text(
        "from typing import Dict, List\n\n"
        "_CORE_CONCEPTS: Dict[str, List[str]] = {\n"
        "    'conflict': ['fight'],\n"
        "    'travel': ['journey'],\n"
        "    'love': ['heart'],\n"
        "}\n",
        encoding="utf-8",
    )
    scoring_path.write_text(
        "from typing import Dict\n\n"
        "_NARRATIVE_WEIGHTS: Dict[str, float] = {\n"
        "    'actions': 0.3,\n"
        "    'emotions': 0.2,\n"
        "}\n\n"
        "_BLEND_WEIGHTS: Dict[str, float] = {\n"
        "    'narrative': 0.25,\n"
        "}\n",
        encoding="utf-8",
    )
    narrative_path.write_text(
        "from typing import Dict, Set\n\n"
        "_SCORING_GROUPS: Dict[str, Set[str]] = {\n"
        "    'actions': {'conflict', 'travel'},\n"
        "    'emotions': {'love'},\n"
        "}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(concept_manager, "CONCEPTS", concepts_path)
    monkeypatch.setattr(concept_manager, "SCORING", scoring_path)
    monkeypatch.setattr(concept_manager, "NARRATIVE", narrative_path)

    original_registry = _snapshot_runtime_concepts()
    original_narrative_weights = dict(scoring._NARRATIVE_WEIGHTS)
    original_blend_weights = dict(scoring._BLEND_WEIGHTS)
    original_groups = {
        dim: sorted(values)
        for dim, values in narrative_elements._SCORING_GROUPS.items()
    }

    try:
        result = concept_manager._delete_concept_group("Conflict")

        assert result == {
            "deleted": True,
            "affected_dimensions": 1,
            "removed_references": 1,
        }
        assert concept_manager._load_concepts() == {
            "travel": ["journey"],
            "love": ["heart"],
        }
        assert concept_manager._load_scoring_groups() == {
            "actions": ["travel"],
            "emotions": ["love"],
        }
        assert concepts._registry == {
            "travel": {"journey"},
            "love": {"heart"},
        }
        assert narrative_elements._SCORING_GROUPS == {
            "actions": {"travel"},
            "emotions": {"love"},
        }
    finally:
        _restore_runtime_concepts(original_registry)
        concept_manager._refresh_runtime_scoring_state(
            original_narrative_weights,
            original_blend_weights,
            original_groups,
        )


def test_preview_score_discards_stale_runtime_concepts_and_narrative_groups():
    temp_group = "preview_temp_group"
    query_text = "alphafoo"
    candidate_text = "betabar"

    try:
        assert concept_manager._preview_score(query_text, candidate_text) == 0.0

        concepts._registry[temp_group] = {"alphafoo", "betabar"}
        concepts._clear_runtime_concept_caches()
        narrative_elements._SCORING_GROUPS["actions"].add(temp_group)
        narrative_elements._extract_narrative_elements.cache_clear()

        stale_runtime_score = _score_with_loaded_modules(query_text, candidate_text)
        reloaded_score = concept_manager._preview_score(query_text, candidate_text)

        assert stale_runtime_score > 0.0
        assert reloaded_score == 0.0
        assert temp_group not in concepts._registry
        assert temp_group not in narrative_elements._SCORING_GROUPS["actions"]
    finally:
        concept_manager._preview_score(query_text, candidate_text)


def test_zero_match_query_returns_empty_list():
    history = [
        {"role": "user", "content": "Alice discussed Python testing."},
        {"role": "assistant", "content": "Python tests catch regressions."},
        {"role": "user", "content": "Bob deployed the service."},
        {"role": "assistant", "content": "Deployment finished successfully."},
        {"role": "user", "content": "Recent chat to exclude."},
        {"role": "assistant", "content": "Another recent chat to exclude."},
    ]

    results = retrieval.retrieve_related_messages(
        history,
        query_text="quantum teleportation",
        exclude_last_n=2,
        importance_weight=0.0,
    )

    assert results == []


def test_register_and_unregister_concepts_normalize_group_name_casing():
    normalized_group = "mixedcaseruntime"

    concepts.unregister_concepts(normalized_group)
    try:
        concepts.register_concepts(" MixedCaseRuntime ", ["Token"])

        assert normalized_group in concepts.list_concept_groups()
        assert concepts.get_word_to_groups()["token"] == frozenset({normalized_group})
        assert concepts.unregister_concepts("MIXEDCASERUNTIME") is True
        assert normalized_group not in concepts.list_concept_groups()
    finally:
        concepts.unregister_concepts(normalized_group)


def test_sentence_starter_words_not_extracted_as_entities():
    elements = narrative_elements._extract_narrative_elements("Can you help Alice with Python?")

    assert "Can" not in elements["entities"]
    assert "Alice" in elements["entities"]


def test_happiness_matches_happy_via_stemmer():
    happiness_tokens = _tokenize("happiness")
    happy_tokens = _tokenize("happy")

    assert set(happiness_tokens) & set(happy_tokens) == {"happy"}
    assert _score_with_loaded_modules("happiness", "I feel happy.") > 0.0


def test_importance_weight_out_of_range_raises_value_error():
    history = [
        {"role": "assistant", "content": "Python web framework comparison details."},
        {"role": "user", "content": "recent"},
    ]

    for invalid_weight in (-0.1, 1.1):
        with pytest.raises(ValueError, match="importance_weight"):
            retrieval.retrieve_related_messages(
                history,
                query_text="Python web framework comparison",
                exclude_last_n=1,
                importance_weight=invalid_weight,
            )


def test_importance_weight_boundary_values_preserve_expected_match():
    history = [
        {
            "role": "assistant",
            "content": "Alice asked about Python web frameworks and Django deployment details?",
        },
        {"role": "assistant", "content": "random note"},
        {"role": "user", "content": "recent"},
    ]

    for weight in (0.0, 1.0):
        results = retrieval.retrieve_related_messages(
            history,
            query_text="Python Django web framework",
            exclude_last_n=1,
            importance_weight=weight,
        )

        assert results == [
            {
                "role": "assistant",
                "content": "[RELATED_MEMORY] Alice asked about Python web frameworks and Django deployment details?",
            }
        ]


@pytest.mark.parametrize(
    ("parameter_name", "kwargs"),
    [
        ("max_retrieved", {"max_retrieved": -1}),
        ("exclude_last_n", {"exclude_last_n": -1}),
        ("target_token_count", {"enable_token_based_retrieval": True, "target_token_count": -1}),
        ("current_token_count", {"enable_token_based_retrieval": True, "current_token_count": -1}),
        (
            "max_retrieved_for_token_target",
            {"enable_token_based_retrieval": True, "max_retrieved_for_token_target": -1},
        ),
    ],
)
def test_retrieval_negative_integer_parameters_raise_value_error(parameter_name, kwargs):
    history = [
        {"role": "assistant", "content": "Python web framework comparison details."},
        {"role": "user", "content": "recent"},
    ]
    request_kwargs = {
        "exclude_last_n": 1,
        "importance_weight": 0.0,
    }
    request_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=parameter_name):
        retrieval.retrieve_related_messages(
            history,
            query_text="Python web framework comparison",
            **request_kwargs,
        )


def test_retrieval_zero_max_retrieved_returns_empty_list():
    history = [
        {"role": "assistant", "content": "alpha beta"},
        {"role": "user", "content": "recent"},
    ]

    results = retrieval.retrieve_related_messages(
        history,
        query_text="alpha beta",
        max_retrieved=0,
        exclude_last_n=1,
        enable_clustering=False,
        importance_weight=0.0,
    )

    assert results == []


def test_create_memory_index_rejects_zero_chunk_size():
    with pytest.raises(ValueError, match="chunk_size"):
        create_memory_index([], chunk_size=0)


def test_score_detailed_with_weights_respects_explicit_empty_maps():
    default_score = _score_with_loaded_modules(
        "python web framework comparison",
        "Python web frameworks are useful for backend apps.",
    )
    empty_override_score = _score_with_loaded_modules_and_weights(
        "python web framework comparison",
        "Python web frameworks are useful for backend apps.",
        narrative_weights={},
        blend_weights={},
    )

    assert default_score > 0.0
    assert empty_override_score == 0.0


def test_score_detailed_honors_runtime_weight_changes_without_manual_cache_clear():
    query_text = "python web framework comparison"
    candidate_text = "Python web frameworks are useful for backend apps."
    original_blend_weights = dict(scoring._BLEND_WEIGHTS)

    try:
        scoring._score_detailed.cache_clear()
        baseline_score = _score_with_loaded_modules(query_text, candidate_text)

        for key in scoring._BLEND_WEIGHTS:
            scoring._BLEND_WEIGHTS[key] = 0.0

        changed_score = _score_with_loaded_modules(query_text, candidate_text)

        assert baseline_score > 0.0
        assert changed_score == 0.0
    finally:
        scoring._BLEND_WEIGHTS.clear()
        scoring._BLEND_WEIGHTS.update(original_blend_weights)
        scoring._score_detailed.cache_clear()


def test_retrieval_reflects_runtime_weight_changes_without_manual_cache_clear():
    history = [
        {"role": "assistant", "content": "Python web frameworks are useful for backend apps."},
        {"role": "user", "content": "recent"},
    ]
    original_blend_weights = dict(scoring._BLEND_WEIGHTS)

    try:
        scoring._score_detailed.cache_clear()
        before = retrieval.retrieve_related_messages(
            history,
            query_text="python web framework comparison",
            exclude_last_n=1,
            enable_clustering=False,
            importance_weight=0.0,
        )

        for key in scoring._BLEND_WEIGHTS:
            scoring._BLEND_WEIGHTS[key] = 0.0

        after = retrieval.retrieve_related_messages(
            history,
            query_text="python web framework comparison",
            exclude_last_n=1,
            enable_clustering=False,
            importance_weight=0.0,
        )

        assert before == [
            {
                "role": "assistant",
                "content": "[RELATED_MEMORY] Python web frameworks are useful for backend apps.",
            }
        ]
        assert after == []
    finally:
        scoring._BLEND_WEIGHTS.clear()
        scoring._BLEND_WEIGHTS.update(original_blend_weights)
        scoring._score_detailed.cache_clear()


def test_preview_score_uses_unsaved_scoring_payload_without_writing_files():
    zero_blend_weights = {key: 0.0 for key in scoring._BLEND_WEIGHTS}
    zero_narrative_weights = {key: 0.0 for key in scoring._NARRATIVE_WEIGHTS}

    baseline_score = concept_manager._preview_score(
        "king",
        "queen",
        scoring_payload={
            "_BLEND_WEIGHTS": {**zero_blend_weights, "narrative": 1.0},
            "_NARRATIVE_WEIGHTS": {**zero_narrative_weights, "actions": 1.0},
        },
        groups_payload={"actions": []},
    )
    preview_score = concept_manager._preview_score(
        "king",
        "queen",
        scoring_payload={
            "_BLEND_WEIGHTS": {**zero_blend_weights, "narrative": 1.0},
            "_NARRATIVE_WEIGHTS": {**zero_narrative_weights, "actions": 1.0},
        },
        groups_payload={"actions": ["authority"]},
    )

    assert baseline_score == 0.0
    assert preview_score == 1.0
    assert "authority" not in narrative_elements._SCORING_GROUPS.get("actions", set())


def test_preview_score_respects_explicit_empty_scoring_maps():
    score = concept_manager._preview_score(
        "python web framework comparison",
        "Python web frameworks are useful for backend apps.",
        scoring_payload={"_BLEND_WEIGHTS": {}, "_NARRATIVE_WEIGHTS": {}},
        groups_payload={"actions": ["conflict"]},
    )

    assert score == 0.0


def test_preview_score_accepts_valid_concept_group_ids():
    zero_blend_weights = {key: 0.0 for key in scoring._BLEND_WEIGHTS}
    zero_narrative_weights = {key: 0.0 for key in scoring._NARRATIVE_WEIGHTS}

    score = concept_manager._preview_score(
        "They fight at dawn.",
        "They fight again at dawn.",
        scoring_payload={
            "_BLEND_WEIGHTS": {**zero_blend_weights, "narrative": 1.0},
            "_NARRATIVE_WEIGHTS": {**zero_narrative_weights, "actions": 1.0},
        },
        groups_payload={"actions": ["conflict"]},
    )

    assert score == 1.0


def test_retrieval_tolerates_none_message_content():
    history = [
        {"role": "assistant", "content": None},
        {"role": "assistant", "content": "alpha beta"},
        {"role": "user", "content": "recent"},
    ]

    results = retrieval.retrieve_related_messages(
        history,
        query_text="alpha beta",
        exclude_last_n=1,
        enable_clustering=False,
        importance_weight=0.0,
    )

    assert results == [
        {
            "role": "assistant",
            "content": "[RELATED_MEMORY] alpha beta",
        }
    ]


def test_analysis_helpers_tolerate_none_message_content():
    history = [
        {"role": "assistant", "content": None},
        {"role": "assistant", "content": "Alice mentioned Python testing."},
    ]

    summary = summarize_conversation_window(history)
    stats = get_memory_statistics(history)
    index = create_memory_index(history, chunk_size=1)

    assert "Alice" in summary
    assert stats["total_messages"] == 2
    assert stats["entities"] == ["Alice", "Python"]
    assert len(index) == 2


def test_save_scoring_bundle_refreshes_live_runtime_state(monkeypatch):
    temp_dir = _prepare_temp_dir("save_scoring_bundle_refresh_runtime")
    scoring_path = temp_dir / "scoring.py"
    narrative_path = temp_dir / "narrative_elements.py"
    scoring_path.write_text(
        "from typing import Dict\n\n"
        "_NARRATIVE_WEIGHTS: Dict[str, float] = {\n"
        "    'actions': 0.3,\n"
        "}\n\n"
        "_BLEND_WEIGHTS: Dict[str, float] = {\n"
        "    'tfidf': 0.2,\n"
        "    'narrative': 0.25,\n"
        "}\n",
        encoding="utf-8",
    )
    narrative_path.write_text(
        "from typing import Dict, Set\n\n"
        "_SCORING_GROUPS: Dict[str, Set[str]] = {\n"
        "    'actions': {'meeting'},\n"
        "}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(concept_manager, "SCORING", scoring_path)
    monkeypatch.setattr(concept_manager, "NARRATIVE", narrative_path)

    original_narrative_weights = dict(scoring._NARRATIVE_WEIGHTS)
    original_blend_weights = dict(scoring._BLEND_WEIGHTS)
    original_groups = {
        dim: sorted(values)
        for dim, values in narrative_elements._SCORING_GROUPS.items()
    }

    try:
        concept_manager._save_scoring_bundle(
            {
                "_NARRATIVE_WEIGHTS": {"actions": 0.7},
                "_BLEND_WEIGHTS": {"tfidf": 0.9},
            },
            {"actions": ["authority"]},
        )

        assert scoring._NARRATIVE_WEIGHTS == {"actions": 0.7}
        assert scoring._BLEND_WEIGHTS == {"tfidf": 0.9, "narrative": 0.25}
        assert narrative_elements._SCORING_GROUPS == {"actions": {"authority"}}
    finally:
        concept_manager._refresh_runtime_scoring_state(
            original_narrative_weights,
            original_blend_weights,
            original_groups,
        )


def test_concept_manager_returns_400_for_invalid_preview_group_ids(monkeypatch):
    temp_dir = _prepare_temp_dir("invalid_preview_groups")
    concepts_path = temp_dir / "concepts.py"
    concepts_path.write_text(
        "from typing import Dict, List\n\n"
        "_CORE_CONCEPTS: Dict[str, List[str]] = {\n"
        "    'conflict': ['fight'],\n"
        "    'travel': ['journey'],\n"
        "}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(concept_manager, "CONCEPTS", concepts_path)

    server = HTTPServer(("127.0.0.1", 0), concept_manager.Handler)
    worker = threading.Thread(target=server.handle_request)
    worker.start()

    conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
    try:
        conn.request(
            "POST",
            "/api/preview_score",
            body=json.dumps(
                {
                    "query": "They fight at dawn.",
                    "candidate": "They fight again.",
                    "groups": {"actions": ["fight"]},
                }
            ),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
    finally:
        conn.close()
        server.server_close()
        worker.join(timeout=5)

    assert response.status == 400
    assert "fight" in payload["error"]


def test_concept_manager_returns_400_for_invalid_save_group_ids(monkeypatch):
    temp_dir = _prepare_temp_dir("invalid_save_groups")
    concepts_path = temp_dir / "concepts.py"
    scoring_path = temp_dir / "scoring.py"
    narrative_path = temp_dir / "narrative_elements.py"
    concepts_path.write_text(
        "from typing import Dict, List\n\n"
        "_CORE_CONCEPTS: Dict[str, List[str]] = {\n"
        "    'conflict': ['fight'],\n"
        "    'travel': ['journey'],\n"
        "}\n",
        encoding="utf-8",
    )
    scoring_path.write_text(
        "from typing import Dict\n\n"
        "_NARRATIVE_WEIGHTS: Dict[str, float] = {\n"
        "    'actions': 0.3,\n"
        "}\n\n"
        "_BLEND_WEIGHTS: Dict[str, float] = {\n"
        "    'narrative': 0.25,\n"
        "}\n",
        encoding="utf-8",
    )
    narrative_path.write_text(
        "from typing import Dict, Set\n\n"
        "_SCORING_GROUPS: Dict[str, Set[str]] = {\n"
        "    'actions': {'conflict'},\n"
        "}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(concept_manager, "CONCEPTS", concepts_path)
    monkeypatch.setattr(concept_manager, "SCORING", scoring_path)
    monkeypatch.setattr(concept_manager, "NARRATIVE", narrative_path)

    server = HTTPServer(("127.0.0.1", 0), concept_manager.Handler)
    worker = threading.Thread(target=server.handle_request)
    worker.start()

    conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
    try:
        conn.request(
            "POST",
            "/api/save_scoring_bundle",
            body=json.dumps(
                {
                    "scoring": {"_NARRATIVE_WEIGHTS": {"actions": 0.5}},
                    "groups": {"actions": ["fight"]},
                }
            ),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
    finally:
        conn.close()
        server.server_close()
        worker.join(timeout=5)

    assert response.status == 400
    assert "fight" in payload["error"]


def test_delete_group_api_reports_scoring_cleanup(monkeypatch):
    temp_dir = _prepare_temp_dir("delete_group_api_cleanup")
    concepts_path = temp_dir / "concepts.py"
    scoring_path = temp_dir / "scoring.py"
    narrative_path = temp_dir / "narrative_elements.py"
    concepts_path.write_text(
        "from typing import Dict, List\n\n"
        "_CORE_CONCEPTS: Dict[str, List[str]] = {\n"
        "    'conflict': ['fight'],\n"
        "    'travel': ['journey'],\n"
        "}\n",
        encoding="utf-8",
    )
    scoring_path.write_text(
        "from typing import Dict\n\n"
        "_NARRATIVE_WEIGHTS: Dict[str, float] = {\n"
        "    'actions': 0.3,\n"
        "}\n\n"
        "_BLEND_WEIGHTS: Dict[str, float] = {\n"
        "    'narrative': 0.25,\n"
        "}\n",
        encoding="utf-8",
    )
    narrative_path.write_text(
        "from typing import Dict, Set\n\n"
        "_SCORING_GROUPS: Dict[str, Set[str]] = {\n"
        "    'actions': {'conflict', 'travel'},\n"
        "}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(concept_manager, "CONCEPTS", concepts_path)
    monkeypatch.setattr(concept_manager, "SCORING", scoring_path)
    monkeypatch.setattr(concept_manager, "NARRATIVE", narrative_path)

    original_registry = _snapshot_runtime_concepts()
    original_narrative_weights = dict(scoring._NARRATIVE_WEIGHTS)
    original_blend_weights = dict(scoring._BLEND_WEIGHTS)
    original_groups = {
        dim: sorted(values)
        for dim, values in narrative_elements._SCORING_GROUPS.items()
    }

    server = HTTPServer(("127.0.0.1", 0), concept_manager.Handler)
    worker = threading.Thread(target=server.handle_request)
    worker.start()

    try:
        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        try:
            conn.request(
                "POST",
                "/api/delete_group",
                body=json.dumps({"group": "conflict"}),
                headers={"Content-Type": "application/json"},
            )
            response = conn.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
        finally:
            conn.close()
    finally:
        server.server_close()
        worker.join(timeout=5)
        _restore_runtime_concepts(original_registry)
        concept_manager._refresh_runtime_scoring_state(
            original_narrative_weights,
            original_blend_weights,
            original_groups,
        )

    assert response.status == 200
    assert payload == {
        "ok": True,
        "deleted": True,
        "affected_dimensions": 1,
        "removed_references": 1,
    }
    assert concept_manager._load_scoring_groups() == {
        "actions": ["travel"],
    }
