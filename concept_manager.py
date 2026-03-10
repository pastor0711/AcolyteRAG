"""
A lightweight standalone local server (GUI) allowing users to visually manage 
concept registries, tweak scoring weights, and preview match results in real-time.
"""
import ast
import importlib
import json
import math
import re
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Timer
from typing import Dict, Tuple
from urllib.parse import urlparse

ROOT      = Path(__file__).parent
CONCEPTS  = ROOT / "concepts.py"
SCORING   = ROOT / "scoring.py"
NARRATIVE = ROOT / "narrative_elements.py"
PORT      = 7842


class _BadRequestError(ValueError):
    """Raised when an API request body is syntactically valid but unusable."""


def _reject_non_finite_json_constant(value: str):
    raise _BadRequestError(f"Invalid numeric constant: {value}.")


def _require_mapping(value, field_name: str) -> dict:
    if not isinstance(value, dict):
        raise _BadRequestError(f"{field_name} must be an object.")
    return value


def _require_field(payload: dict, field_name: str):
    if field_name not in payload:
        raise _BadRequestError(f"Missing required field: {field_name}.")
    return payload[field_name]


def _require_string(value, field_name: str) -> str:
    if not isinstance(value, str):
        raise _BadRequestError(f"{field_name} must be a string.")
    return value


def _normalize_string_list(values, field_name: str) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple, set)):
        raise _BadRequestError(f"{field_name} must be an array of strings.")

    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise _BadRequestError(f"{field_name} items must be strings.")
        normalized_value = value.strip().lower()
        if normalized_value and normalized_value not in seen:
            seen.add(normalized_value)
            normalized.append(normalized_value)
    return normalized


def _normalize_concept_group_name(group_name: str) -> str:
    return str(group_name).strip().lower()


def _normalize_concept_words(words) -> list[str]:
    return _normalize_string_list(words, "words")


def _normalize_concepts_payload(concepts: dict) -> dict:
    normalized: Dict[str, list[str]] = {}
    for group, words in concepts.items():
        if not isinstance(words, (list, tuple, set)):
            continue

        group_name = _normalize_concept_group_name(group)
        if not group_name:
            continue

        existing_words = normalized.setdefault(group_name, [])
        seen_words = set(existing_words)
        for word in _normalize_concept_words(words):
            if word not in seen_words:
                seen_words.add(word)
                existing_words.append(word)
    return normalized


def _ensure_package_root_on_path() -> None:
    pkg_root = str(ROOT.parent)
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)


def _load_runtime_modules(reload_modules: bool = False) -> dict:
    _ensure_package_root_on_path()
    importlib.invalidate_caches()

    modules = {
        "text_processing": importlib.import_module("acolyterag.text_processing"),
        "concepts": importlib.import_module("acolyterag.concepts"),
        "concept_expansion": importlib.import_module("acolyterag.concept_expansion"),
        "narrative_elements": importlib.import_module("acolyterag.narrative_elements"),
        "scoring": importlib.import_module("acolyterag.scoring"),
        "retrieval": importlib.import_module("acolyterag.retrieval"),
        "analysis": importlib.import_module("acolyterag.analysis"),
    }

    if reload_modules:
        modules["concepts"] = importlib.reload(modules["concepts"])
        modules["text_processing"] = importlib.reload(modules["text_processing"])
        modules["concept_expansion"] = importlib.reload(modules["concept_expansion"])
        modules["narrative_elements"] = importlib.reload(modules["narrative_elements"])
        modules["scoring"] = importlib.reload(modules["scoring"])

    return modules


def _clear_runtime_scoring_caches(modules: dict) -> None:
    for module_name, attr_name in (
        ("scoring", "_score_detailed"),
        ("retrieval", "_score_detailed"),
        ("narrative_elements", "_extract_narrative_elements"),
        ("retrieval", "_extract_narrative_elements"),
        ("analysis", "_extract_narrative_elements"),
    ):
        cached_fn = getattr(modules[module_name], attr_name, None)
        if cached_fn is not None and hasattr(cached_fn, "cache_clear"):
            cached_fn.cache_clear()


def _apply_runtime_concept_state(modules: dict, concepts: dict) -> None:
    concepts_module = modules["concepts"]

    concepts_module._registry.clear()
    concepts_module._registry.update({
        group: set(words)
        for group, words in concepts.items()
    })

    clear_caches = getattr(concepts_module, "_clear_runtime_concept_caches", None)
    if callable(clear_caches):
        clear_caches()


def _apply_runtime_scoring_state(modules: dict, narrative: dict, blend: dict, groups: dict) -> None:
    scoring = modules["scoring"]
    narrative_elements = modules["narrative_elements"]

    scoring._NARRATIVE_WEIGHTS.clear()
    scoring._NARRATIVE_WEIGHTS.update(narrative)
    scoring._BLEND_WEIGHTS.clear()
    scoring._BLEND_WEIGHTS.update(blend)

    narrative_elements._SCORING_GROUPS.clear()
    narrative_elements._SCORING_GROUPS.update({
        dim: set(words)
        for dim, words in groups.items()
    })

    _clear_runtime_scoring_caches(modules)


def _refresh_runtime_concept_state(concepts: dict) -> None:
    modules = _load_runtime_modules(reload_modules=False)
    _apply_runtime_concept_state(modules, concepts)


def _refresh_runtime_scoring_state(narrative: dict, blend: dict, groups: dict) -> None:
    modules = _load_runtime_modules(reload_modules=False)
    _apply_runtime_scoring_state(modules, narrative, blend, groups)


def _preview_score(
    query_text: str,
    candidate_text: str,
    scoring_payload: dict | None = None,
    groups_payload: dict | None = None,
) -> float:
    """
    Reloads the scoring stack from disk before computing a preview score.
    This keeps the live preview aligned with recently edited concepts and weights.
    """
    modules = _load_runtime_modules(reload_modules=True)
    text_processing = modules["text_processing"]
    concept_expansion = modules["concept_expansion"]
    narrative_elements = modules["narrative_elements"]
    scoring = modules["scoring"]

    scoring_payload = {} if scoring_payload is None else _require_mapping(scoring_payload, "scoring")
    groups_payload = None if groups_payload is None else _require_mapping(groups_payload, "groups")
    narrative_weights, blend_weights = _coerce_scoring_weights(scoring_payload)
    preview_groups = (
        _validate_scoring_groups(_load_scoring_groups())
        if groups_payload is None
        else _coerce_scoring_groups(groups_payload)
    )

    q_tokens = text_processing._tokenize(query_text)
    c_tokens = text_processing._tokenize(candidate_text)
    idf = scoring._build_idf([q_tokens, c_tokens])
    idf_key = tuple(sorted(idf.items()))
    q_conc = concept_expansion._expand_concepts(q_tokens)
    c_conc = concept_expansion._expand_concepts(c_tokens)
    q_elems = narrative_elements._extract_narrative_elements_with_groups(query_text, preview_groups)
    c_elems = narrative_elements._extract_narrative_elements_with_groups(candidate_text, preview_groups)
    q_etuple = tuple((k, tuple(v)) for k, v in q_elems.items())
    c_etuple = tuple((k, tuple(v)) for k, v in c_elems.items())
    return scoring._score_detailed_with_weights(
        query_text=query_text,
        candidate_text=candidate_text,
        query_tokens=q_tokens,
        candidate_tokens=c_tokens,
        idf_key=idf_key,
        query_concepts=q_conc,
        candidate_concepts=c_conc,
        query_elements=q_etuple,
        candidate_elements=c_etuple,
        narrative_weights=narrative_weights,
        blend_weights=blend_weights,
    )


def _load_scoring_weights() -> dict:
    """
    Parses scoring.py via AST to safely extract the current runtime values 
    of _NARRATIVE_WEIGHTS and _BLEND_WEIGHTS without importing.
    """
    src  = SCORING.read_text(encoding="utf-8")
    tree = ast.parse(src)
    result = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = [node.targets[0]]
            value   = node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
            value   = node.value
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id in ("_NARRATIVE_WEIGHTS", "_BLEND_WEIGHTS"):
                result[target.id] = ast.literal_eval(value)
    return result


def _write_text_file(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _coerce_finite_float(name: str, value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise _BadRequestError(f"{name} must be a finite number.") from exc
    if not math.isfinite(number):
        raise _BadRequestError(f"{name} must be a finite number.")
    return number


def _coerce_scoring_weights(payload: dict) -> Tuple[dict, dict]:
    payload = _require_mapping(payload, "payload")
    current_weights = _load_scoring_weights()
    narrative = {
        str(k): _coerce_finite_float(f"_NARRATIVE_WEIGHTS.{k}", v)
        for k, v in current_weights.get("_NARRATIVE_WEIGHTS", {}).items()
    }
    blend = {
        str(k): _coerce_finite_float(f"_BLEND_WEIGHTS.{k}", v)
        for k, v in current_weights.get("_BLEND_WEIGHTS", {}).items()
    }

    if "_NARRATIVE_WEIGHTS" in payload:
        narrative_payload = _require_mapping(
            payload.get("_NARRATIVE_WEIGHTS"),
            "_NARRATIVE_WEIGHTS",
        )
        narrative = {} if not narrative_payload else dict(narrative)
        for key, value in narrative_payload.items():
            narrative[str(key)] = _coerce_finite_float(f"_NARRATIVE_WEIGHTS.{key}", value)

    if "_BLEND_WEIGHTS" in payload:
        blend_payload = _require_mapping(
            payload.get("_BLEND_WEIGHTS"),
            "_BLEND_WEIGHTS",
        )
        blend = {} if not blend_payload else dict(blend)
        for key, value in blend_payload.items():
            blend[str(key)] = _coerce_finite_float(f"_BLEND_WEIGHTS.{key}", value)

    return (
        narrative,
        blend,
    )


def _normalize_scoring_groups(groups: dict) -> dict:
    groups = _require_mapping(groups, "groups")
    normalized: Dict[str, list[str]] = {}
    for dim, words in groups.items():
        dim_name = str(dim).strip().lower()
        if not dim_name:
            continue
        normalized[dim_name] = _normalize_string_list(words, f"groups.{dim_name}")
    return normalized


def _validate_scoring_groups(groups: dict) -> dict:
    valid_group_ids = set(_load_concepts())
    return _validate_scoring_groups_against(groups, valid_group_ids)


def _validate_scoring_groups_against(groups: dict, valid_group_ids: set[str]) -> dict:
    invalid_by_dimension: Dict[str, list[str]] = {}

    for dim, words in groups.items():
        invalid_words = [word for word in words if word not in valid_group_ids]
        if invalid_words:
            invalid_by_dimension[dim] = sorted(invalid_words)

    if invalid_by_dimension:
        details = "; ".join(
            f"{dim}: {', '.join(words)}"
            for dim, words in sorted(invalid_by_dimension.items())
        )
        raise _BadRequestError(f"Unknown concept group IDs for {details}.")
    return groups


def _coerce_scoring_groups(groups: dict) -> dict:
    return _validate_scoring_groups(_normalize_scoring_groups(groups))


def _coerce_scoring_groups_against(groups: dict, valid_group_ids: set[str]) -> dict:
    return _validate_scoring_groups_against(_normalize_scoring_groups(groups), valid_group_ids)


def _render_scoring_weights_source(src: str, narrative: dict, blend: dict) -> str:
    def _render_dict(name: str, d: dict) -> str:
        if not d:
            return f"{name}: Dict[str, float] = {{}}"
        items = ",\n".join(f"    {repr(k):20s}: {v}" for k, v in d.items())
        return f"{name}: Dict[str, float] = {{\n{items},\n}}"

    updated = src
    for dict_name, values in [
        ("_NARRATIVE_WEIGHTS", narrative),
        ("_BLEND_WEIGHTS", blend),
    ]:
        pat = re.compile(rf"{dict_name}: Dict\[str, float\] = \{{.*?\}}", re.DOTALL)
        if not pat.search(updated):
            raise ValueError(f"Could not locate {dict_name} in scoring.py")
        updated = pat.sub(_render_dict(dict_name, values), updated)
    return updated


def _save_scoring_weights(narrative: dict, blend: dict) -> None:
    """
    Serializes updated weight dictionaries and uses regex to replace the 
    existing dict definitions within the scoring.py source file.
    """
    src = SCORING.read_text(encoding="utf-8")
    normalized_groups = _validate_scoring_groups(_load_scoring_groups())
    updated = _render_scoring_weights_source(src, narrative, blend)
    try:
        _write_text_file(SCORING, updated)
        _refresh_runtime_scoring_state(narrative, blend, normalized_groups)
    except Exception as exc:
        try:
            _write_text_file(SCORING, src)
        except Exception as rollback_exc:
            raise RuntimeError(f"{exc}; rollback failed: {rollback_exc}") from exc
        raise


def _load_scoring_groups() -> dict:
    """Read _SCORING_GROUPS from narrative_elements.py as {dim: [word, ...]}."""
    src  = NARRATIVE.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and node.value is not None:
            target = node.target
            if isinstance(target, ast.Name) and target.id == "_SCORING_GROUPS":
                raw = ast.literal_eval(node.value)
                return {k: sorted(v) for k, v in raw.items()}
    return {}


def _render_scoring_groups_source(src: str, groups: dict) -> str:
    """Write updated _SCORING_GROUPS back into narrative_elements.py using
    AST to locate the exact line range — immune to formatting/corruption."""
    groups = _normalize_scoring_groups(groups)
    file_lines = src.splitlines(keepends=True)

    # ── locate the node via AST ──────────────────────────────────────────────
    try:
        tree = ast.parse(src)
    except SyntaxError:
        # File is already broken; fall back to a header-scan replacement.
        tree = None

    start_line = end_line = None   # 0-indexed

    if tree:
        for node in ast.walk(tree):
            is_ann = (
                isinstance(node, ast.AnnAssign)
                and node.value is not None
                and isinstance(node.target, ast.Name)
                and node.target.id == "_SCORING_GROUPS"
            )
            if is_ann:
                # AST lines are 1-indexed; convert to 0-indexed slice bounds.
                start_line = node.col_offset   # always 0 for module-level
                start_line = node.lineno - 1
                end_line   = node.end_lineno   # exclusive upper bound (0-idx)
                break

    if start_line is None:
        # Fallback: find the header line by scanning
        for i, line in enumerate(file_lines):
            if line.lstrip().startswith("_SCORING_GROUPS: Dict[str, Set[str]] ="):
                start_line = i
                # Walk forward until a line that is solely "}" or "},"
                for j in range(i + 1, len(file_lines)):
                    stripped = file_lines[j].strip()
                    if stripped in ("}", "},"):
                        end_line = j + 1
                        break
                break

    if start_line is None:
        raise ValueError("Could not locate _SCORING_GROUPS in narrative_elements.py")

    # ── render new block ─────────────────────────────────────────────────────
    def _render_set(words) -> str:
        if not words:
            return "set()"
        items = ", ".join(repr(w) for w in sorted(words))
        return "{" + items + "}"

    block_lines = ["_SCORING_GROUPS: Dict[str, Set[str]] = {"]
    for dim, words in groups.items():
        block_lines.append(f"    {repr(dim):20s}: {_render_set(words)},")
    block_lines.append("}")
    rendered = "\n".join(block_lines) + "\n"

    # ── splice and write ─────────────────────────────────────────────────────
    new_lines = file_lines[:start_line] + [rendered] + file_lines[end_line:]
    return "".join(new_lines)


def _save_scoring_groups(groups: dict) -> None:
    normalized_groups = _coerce_scoring_groups(groups)
    current_weights = _load_scoring_weights()
    src = NARRATIVE.read_text(encoding="utf-8")
    updated = _render_scoring_groups_source(src, normalized_groups)
    try:
        _write_text_file(NARRATIVE, updated)
        _refresh_runtime_scoring_state(
            current_weights.get("_NARRATIVE_WEIGHTS", {}),
            current_weights.get("_BLEND_WEIGHTS", {}),
            normalized_groups,
        )
    except Exception as exc:
        try:
            _write_text_file(NARRATIVE, src)
        except Exception as rollback_exc:
            raise RuntimeError(f"{exc}; rollback failed: {rollback_exc}") from exc
        raise


def _save_scoring_bundle(scoring_payload: dict, groups_payload: dict) -> None:
    narrative, blend = _coerce_scoring_weights(scoring_payload)
    normalized_groups = _coerce_scoring_groups(groups_payload)
    current_weights = _load_scoring_weights()
    current_groups = _load_scoring_groups()

    scoring_src = SCORING.read_text(encoding="utf-8")
    narrative_src = NARRATIVE.read_text(encoding="utf-8")
    updated_scoring = _render_scoring_weights_source(scoring_src, narrative, blend)
    updated_narrative = _render_scoring_groups_source(narrative_src, normalized_groups)

    try:
        _write_text_file(SCORING, updated_scoring)
        _write_text_file(NARRATIVE, updated_narrative)
        _refresh_runtime_scoring_state(narrative, blend, normalized_groups)
    except Exception as exc:
        try:
            _write_text_file(SCORING, scoring_src)
            _write_text_file(NARRATIVE, narrative_src)
            modules = _load_runtime_modules(reload_modules=False)
            _apply_runtime_scoring_state(
                modules,
                current_weights.get("_NARRATIVE_WEIGHTS", {}),
                current_weights.get("_BLEND_WEIGHTS", {}),
                current_groups,
            )
        except Exception as rollback_exc:
            raise RuntimeError(f"{exc}; rollback failed: {rollback_exc}") from exc
        raise


def _load_concepts() -> dict:
    src  = CONCEPTS.read_text(encoding="utf-8")
    tree = ast.parse(src)
    concepts = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
            value   = node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
            value   = node.value
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id == "_CORE_CONCEPTS":
                for key, val in zip(value.keys, value.values):
                    group = ast.literal_eval(key)
                    words = ast.literal_eval(val)
                    concepts[group] = list(dict.fromkeys(words))
    return _normalize_concepts_payload(concepts)


def _render_core_concepts_source(src: str, concepts: dict) -> str:
    concepts = _normalize_concepts_payload(concepts)
    block   = _render_core_concepts(concepts)
    pattern = re.compile(r"_CORE_CONCEPTS:\s*Dict\[.*?\]\s*=\s*\{.*?\n\}", re.DOTALL)
    if not pattern.search(src):
        raise ValueError("Could not locate _CORE_CONCEPTS in concepts.py")
    return pattern.sub(block, src)


def _save_concepts(concepts: dict) -> None:
    normalized_concepts = _normalize_concepts_payload(concepts)
    src = CONCEPTS.read_text(encoding="utf-8")
    updated = _render_core_concepts_source(src, normalized_concepts)
    try:
        _write_text_file(CONCEPTS, updated)
        _refresh_runtime_concept_state(normalized_concepts)
    except Exception as exc:
        try:
            _write_text_file(CONCEPTS, src)
        except Exception as rollback_exc:
            raise RuntimeError(f"{exc}; rollback failed: {rollback_exc}") from exc
        raise


def _remove_concept_group_references(groups: dict, group: str) -> Tuple[dict, int, int]:
    cleaned_groups = _normalize_scoring_groups(groups)
    normalized_group = _normalize_concept_group_name(group)
    affected_dimensions = 0
    removed_references = 0

    for dim, group_ids in cleaned_groups.items():
        filtered_group_ids = [group_id for group_id in group_ids if group_id != normalized_group]
        removed_count = len(group_ids) - len(filtered_group_ids)
        if removed_count:
            cleaned_groups[dim] = filtered_group_ids
            affected_dimensions += 1
            removed_references += removed_count

    return cleaned_groups, affected_dimensions, removed_references


def _delete_concept_group(group: str) -> dict:
    normalized_group = _normalize_concept_group_name(group)
    original_concepts = _load_concepts()
    concepts = _normalize_concepts_payload(original_concepts)
    deleted = normalized_group in concepts
    concepts.pop(normalized_group, None)

    current_weights = _load_scoring_weights()
    scoring_groups = _load_scoring_groups()
    cleaned_groups, affected_dimensions, removed_references = _remove_concept_group_references(
        scoring_groups,
        normalized_group,
    )
    cleaned_groups = _coerce_scoring_groups_against(cleaned_groups, set(concepts))

    concepts_changed = deleted
    scoring_changed = removed_references > 0

    if not concepts_changed and not scoring_changed:
        return {
            "deleted": deleted,
            "affected_dimensions": affected_dimensions,
            "removed_references": removed_references,
        }

    concepts_src = CONCEPTS.read_text(encoding="utf-8") if concepts_changed else ""
    narrative_src = NARRATIVE.read_text(encoding="utf-8") if scoring_changed else ""
    updated_concepts = (
        _render_core_concepts_source(concepts_src, concepts)
        if concepts_changed else ""
    )
    updated_narrative = (
        _render_scoring_groups_source(narrative_src, cleaned_groups)
        if scoring_changed else ""
    )

    try:
        if concepts_changed:
            _write_text_file(CONCEPTS, updated_concepts)
        if scoring_changed:
            _write_text_file(NARRATIVE, updated_narrative)
        if concepts_changed:
            _refresh_runtime_concept_state(concepts)
        if scoring_changed:
            _refresh_runtime_scoring_state(
                current_weights.get("_NARRATIVE_WEIGHTS", {}),
                current_weights.get("_BLEND_WEIGHTS", {}),
                cleaned_groups,
            )
    except Exception as exc:
        try:
            if concepts_changed:
                _write_text_file(CONCEPTS, concepts_src)
            if scoring_changed:
                _write_text_file(NARRATIVE, narrative_src)
            modules = _load_runtime_modules(reload_modules=False)
            if concepts_changed:
                _apply_runtime_concept_state(modules, original_concepts)
            if scoring_changed:
                _apply_runtime_scoring_state(
                    modules,
                    current_weights.get("_NARRATIVE_WEIGHTS", {}),
                    current_weights.get("_BLEND_WEIGHTS", {}),
                    scoring_groups,
                )
        except Exception as rollback_exc:
            raise RuntimeError(f"{exc}; rollback failed: {rollback_exc}") from exc
        raise

    return {
        "deleted": deleted,
        "affected_dimensions": affected_dimensions,
        "removed_references": removed_references,
    }


def _render_core_concepts(concepts: dict) -> str:
    lines = ["_CORE_CONCEPTS: Dict[str, List[str]] = {"]
    for group, words in concepts.items():
        word_reprs = [repr(w) for w in words]
        joined     = ", ".join(word_reprs)
        if len(joined) + len(group) + 16 <= 88:
            lines.append(f"    {repr(group):20s}: [{joined}],")
        else:
            chunks, chunk = [], []
            for wr in word_reprs:
                if chunk and sum(len(x) + 2 for x in chunk) + len(wr) > 70:
                    prefix = "                     " if chunks else f"    {repr(group):20s}: ["
                    lines.append(f"{prefix}{', '.join(chunk)},")
                    chunks.append(chunk)
                    chunk = [wr]
                else:
                    chunk.append(wr)
            prefix = "                     " if chunks else f"    {repr(group):20s}: ["
            lines.append(f"{prefix}{', '.join(chunk)}],")
    lines.append("}")
    return "\n".join(lines)


STATIC_DIR = ROOT


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass

    def _send(self, status, ct, body):
        self.send_response(status)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        """Serves static files (HTML/CSS/JS) and responds to data retrieval API fetches."""
        path = urlparse(self.path).path


        _STATIC = {
            "/":              ("manager.html", "text/html; charset=utf-8"),
            "/index.html":    ("manager.html", "text/html; charset=utf-8"),
            "/manager.html":  ("manager.html", "text/html; charset=utf-8"),
            "/manager.css":   ("manager.css",  "text/css; charset=utf-8"),
            "/manager.js":    ("manager.js",   "application/javascript; charset=utf-8"),
        }
        if path in _STATIC:
            fname, ct = _STATIC[path]
            data = (STATIC_DIR / fname).read_bytes()
            self._send(200, ct, data)
            return
        elif path == "/api/concepts":
            self._send(200, "application/json", json.dumps(_load_concepts(), ensure_ascii=False).encode())
        elif path == "/api/get_scoring":
            self._send(200, "application/json", json.dumps(_load_scoring_weights(), ensure_ascii=False).encode())
        elif path == "/api/get_scoring_groups":
            self._send(200, "application/json", json.dumps(_load_scoring_groups(), ensure_ascii=False).encode())
        else:
            self._send(404, "text/plain", b"Not found")

    def do_POST(self):
        """Processes API mutation endpoints (adding words, managing groups, live preview)."""
        length = int(self.headers.get("Content-Length", 0))
        path   = urlparse(self.path).path
        try:
            raw_body = self.rfile.read(length)
            body = (
                json.loads(raw_body, parse_constant=_reject_non_finite_json_constant)
                if raw_body else {}
            )
            if not isinstance(body, dict):
                raise _BadRequestError("JSON body must be an object.")

            concepts = _load_concepts()
            if path == "/api/add_words":
                group = _normalize_concept_group_name(
                    _require_string(_require_field(body, "group"), "group")
                )
                words = _normalize_concept_words(_require_field(body, "words"))
                if group not in concepts:
                    raise _BadRequestError(f"Unknown concept group: {group}.")
                for w in words:
                    if w not in concepts[group]:
                        concepts[group].append(w)
                _save_concepts(concepts)
            elif path == "/api/remove_word":
                group = _normalize_concept_group_name(
                    _require_string(_require_field(body, "group"), "group")
                )
                if group not in concepts:
                    raise _BadRequestError(f"Unknown concept group: {group}.")
                word = _require_string(_require_field(body, "word"), "word").lower().strip()
                concepts[group] = [w for w in concepts[group] if w != word]
                _save_concepts(concepts)
            elif path == "/api/add_group":
                group = _normalize_concept_group_name(
                    _require_string(_require_field(body, "group"), "group")
                )
                if group in concepts:
                    raise _BadRequestError(f"Concept group already exists: {group}.")
                words = _normalize_concept_words(body.get("words", []))
                concepts[group] = list(dict.fromkeys(words))
                _save_concepts(concepts)
            elif path == "/api/delete_group":
                group = _normalize_concept_group_name(
                    _require_string(_require_field(body, "group"), "group")
                )
                result = _delete_concept_group(group)
                self._send(200, "application/json", json.dumps({"ok": True, **result}).encode())
                return
            elif path == "/api/save_scoring":
                narrative, blend = _coerce_scoring_weights(body)
                _save_scoring_weights(narrative, blend)
            elif path == "/api/save_scoring_groups":
                _save_scoring_groups(body)
            elif path == "/api/save_scoring_bundle":
                scoring_payload = _require_mapping(body.get("scoring", {}), "scoring")
                groups_payload = _require_mapping(body.get("groups", {}), "groups")
                _save_scoring_bundle(scoring_payload, groups_payload)
            elif path == "/api/preview_score":
                query_text = _require_string(body.get("query", ""), "query")
                candidate_text = _require_string(body.get("candidate", ""), "candidate")
                scoring_payload = None if "scoring" not in body else _require_mapping(body.get("scoring"), "scoring")
                groups_payload = None if "groups" not in body else _require_mapping(body.get("groups"), "groups")
                score = _preview_score(
                    query_text,
                    candidate_text,
                    scoring_payload=scoring_payload,
                    groups_payload=groups_payload,
                )
                self._send(200, "application/json", json.dumps({"score": score}).encode())
                return
            else:
                self._send(404, "text/plain", b"Not found"); return
            self._send(200, "application/json", b'{"ok":true}')
        except json.JSONDecodeError:
            self._send(
                400,
                "application/json",
                json.dumps({"error": "Invalid JSON payload."}).encode(),
            )
        except _BadRequestError as e:
            self._send(400, "application/json", json.dumps({"error": str(e)}).encode())
        except Exception as e:
            self._send(500, "text/plain", str(e).encode())


def main():
    if not CONCEPTS.exists():
        print(f"ERROR: Cannot find {CONCEPTS}")
        sys.exit(1)
    if not SCORING.exists():
        print(f"ERROR: Cannot find {SCORING}")
        sys.exit(1)
    if not NARRATIVE.exists():
        print(f"ERROR: Cannot find {NARRATIVE}")
        sys.exit(1)
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    url    = f"http://localhost:{PORT}"
    print(f"AcolyteRAG Concept Manager → {url}")
    print("Press Ctrl+C to stop.")
    Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
