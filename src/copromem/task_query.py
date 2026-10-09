"""Public task-query helpers used by CoProMem."""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .intents import verify_intent_registry


POLICY_ID = "copromem-task-query"
_TOKEN = re.compile(r"[a-z][a-z0-9_]{1,}")
_SPLIT = re.compile(r"[._-]+")
_GENERIC_ACTIONS = frozenset({"add", "create", "delete", "get", "remove", "search", "send", "show", "update"})
_NON_SEMANTIC_APPS = frozenset({"api_docs", "supervisor"})
_AUTH_RUNTIME_TERMS = frozenset({"auth", "authenticate", "authentication", "login", "log_in", "logout"})
_HTTP_ACTION_CLASSES = {
    "GET": frozenset({"get", "show", "search", "find", "list", "view", "read"}),
    "POST": frozenset({"create", "add", "send", "attach", "upload", "initiate", "make", "request"}),
    "PATCH": frozenset({"update", "edit", "modify", "change"}),
    "PUT": frozenset({"update", "edit", "modify", "change"}),
    "DELETE": frozenset({"delete", "remove", "clear"}),
}
_INTENT_STOP = frozenset({"a", "an", "the", "to", "for", "of", "in", "on", "with", "and", "or", "from",
                          "your", "you", "my", "i", "this", "that", "all", "each", "any", "via"})
_CURRENCY = re.compile(r"[$€£]\s*\d")
_RECIPIENT = re.compile(r"\b(?:to|for)\s+(?:[A-Z][\w-]*|him|her|them|each|every)\b", re.I)
_PAST_CONTEXT = re.compile(r"\b(?:yesterday|previously|before|ago|last\s+\w+)\b", re.I)
_ACTION_SURFACES = {
    "create": frozenset({"create", "make", "send", "pay", "transfer"}),
    "show": frozenset({"show", "list", "find", "search", "view", "history", "get"}),
    "update": frozenset({"update", "edit", "modify", "change"}),
    "delete": frozenset({"delete", "remove"}),
}
_CONCEPT_SURFACES = {
    "money": frozenset({"money", "payment", "amount", "cash", "transfer"}),
    "transaction": frozenset({"transaction", "payment", "transfer"}),
    "user": frozenset({"user", "person", "people", "recipient", "receiver"}),
    "note": frozenset({"note", "log"}),
}


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _stem(token: str) -> str:
    """A deliberately small, task-independent English surface normalizer."""
    token = token.lower()
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        token = token[:-3]
        return token[:-1] if len(token) > 2 and token[-1:] == token[-2:-1] else token
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(sorted({_stem(item) for item in _TOKEN.findall(text.lower())}))


def _operation_tokens(operation: str) -> tuple[str, str, tuple[str, ...]]:
    # Canonical operation namespace is ``<source>.<domain>.<action>``.
    # Application and callable names can themselves contain underscores; only
    # the two dots delimit namespace fields.  Splitting the full string on
    # underscores would turn ``simple_note`` into the nonexistent app
    # ``simple`` and corrupt public-app compatibility.
    raw = operation.lower().split(".", 2)
    if len(raw) != 3 or not raw[0] or not raw[1] or not raw[2]:
        raise ValueError("registry operation is not canonical")
    app, leaf = raw[1], tuple(_stem(item) for item in _SPLIT.split(raw[2]) if item)
    if not leaf:
        raise ValueError("registry operation has no callable name")
    return app, leaf[0], leaf[1:]


def _operation_index(registry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    registry_hash = registry.get("registry_sha256")
    if not isinstance(registry_hash, str) or not registry_hash:
        raise ValueError("frozen callable registry identity required")
    rows = registry.get("operations")
    if not isinstance(rows, Sequence):
        raise ValueError("frozen callable registry has no operations")
    index: dict[str, Mapping[str, Any]] = {}
    for item in rows:
        if not isinstance(item, Mapping) or not isinstance(item.get("operation"), str):
            raise ValueError("malformed callable registry operation")
        operation = str(item["operation"])
        _operation_tokens(operation)
        if operation in index:
            raise ValueError("duplicate callable registry operation")
        index[operation] = item
    return index


def _aliases(registry: Mapping[str, Any], index: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    raw = registry.get("normalization", {}).get("operation_aliases", {})
    if not isinstance(raw, Mapping):
        raise ValueError("registry operation aliases are malformed")
    aliases = {str(alias): str(operation) for alias, operation in raw.items()}
    if any(operation not in index for operation in aliases.values()):
        raise ValueError("registry alias points to an unknown operation")
    return aliases


def canonical_operation(operation: str, registry: Mapping[str, Any]) -> str:
    index = _operation_index(registry)
    aliases = _aliases(registry, index)
    result = aliases.get(operation, operation)
    if result not in index:
        raise ValueError("operation is not declared by the frozen registry")
    return result


def derive_task_query(instruction: str, domain: str, public_tool_metadata: Mapping[str, Any],
                      callable_registry: Mapping[str, Any]) -> dict[str, Any]:
    """Derive an operation query from public pre-execution information only.

    An operation is supported by a verb plus at least one non-generic callable
    component present in the task instruction.  App names explicitly mentioned
    in the instruction narrow the candidate set; a generic tool catalogue does
    not itself claim that every app is task-relevant.
    """
    if not isinstance(instruction, str) or not isinstance(domain, str):
        raise ValueError("public instruction and domain are required")
    if not isinstance(public_tool_metadata, Mapping):
        raise ValueError("public tool metadata is required")
    index = _operation_index(callable_registry)
    aliases = _aliases(callable_registry, index)
    tokens = set(_tokens(instruction))
    catalog = public_tool_metadata.get("app_descriptions", {})
    if not isinstance(catalog, Mapping):
        raise ValueError("public app descriptions are required")
    public_apps = {str(name).lower() for name in catalog if isinstance(name, str)}
    named_apps = tuple(sorted(app for app in public_apps
                              if all(_stem(part) in tokens for part in app.split("_"))))
    matches: list[dict[str, Any]] = []
    for operation, meta in sorted(index.items()):
        app, verb, nouns = _operation_tokens(operation)
        if app in _NON_SEMANTIC_APPS or verb in _AUTH_RUNTIME_TERMS or any(noun in _AUTH_RUNTIME_TERMS for noun in nouns):
            continue
        if named_apps and app not in named_apps:
            continue
        noun_matches = sorted(set(nouns) & tokens)
        # A bare generic verb (for example, "show") is never enough to infer
        # an API operation.  A callable with only a verb needs an explicit app.
        action_terms = _HTTP_ACTION_CLASSES.get(str(meta.get("http_method") or "").upper(), frozenset({verb}))
        verb_match = verb in tokens or bool(action_terms & tokens)
        specific_nouns = [noun for noun in nouns if noun not in _GENERIC_ACTIONS]
        supported = verb_match and (bool(noun_matches) or (not specific_nouns and app in named_apps))
        if not supported:
            continue
        matches.append({"operation": operation, "app": app, "verb": verb,
                        "matched_nouns": noun_matches, "matched_action_terms": sorted(action_terms & tokens),
                        "match_strength": len(noun_matches) + int(app in named_apps),
                        "public_access_mode": str(meta.get("access_mode") or "unknown")})
    # A named app and complete, unique callable noun identify a terminal even
    # when the user does not phrase the API verb explicitly.
    noun_candidates = []
    for app in named_apps:
        for operation, meta in sorted(index.items()):
            op_app, verb, nouns = _operation_tokens(operation)
            if op_app != app or op_app in _NON_SEMANTIC_APPS or verb in _AUTH_RUNTIME_TERMS:
                continue
            specific = tuple(noun for noun in nouns if noun not in _GENERIC_ACTIONS)
            if not specific or not set(specific) <= tokens:
                continue
            actions = _HTTP_ACTION_CLASSES.get(str(meta.get("http_method") or "").upper(), frozenset({verb}))
            if verb in tokens or actions & tokens:
                continue
            noun_candidates.append((operation, specific, app, verb, meta))
    existing = {item["operation"] for item in matches}
    noun_added = []
    for operation, nouns, app, verb, meta in noun_candidates:
        if sum(other_app == app and other_nouns == nouns for _, other_nouns, other_app, _, _ in noun_candidates) == 1:
            if operation not in existing:
                noun_added.append({"operation": operation, "app": app, "verb": verb,
                                   "matched_nouns": list(nouns), "matched_action_terms": [],
                                   "match_strength": len(nouns) + 1,
                                   "public_access_mode": str(meta.get("access_mode") or "unknown"),
                                   "match_mode": "unique_named_app_complete_noun"})
    matches.extend(noun_added)
    intent_added = []
    intents = public_tool_metadata.get("public_operation_intents")
    if intents is not None:
        if not isinstance(intents, Mapping):
            raise ValueError("public operation intents must be a frozen registry")
        verify_intent_registry(intents, callable_registry)
        intent_tokens = set(tokens)
        if _CURRENCY.search(instruction):
            intent_tokens.update({"money", "amount"})
        if _RECIPIENT.search(instruction):
            intent_tokens.update({"user", "receiver", "recipient"})
        candidates = []
        for row in intents["operations"]:
            operation = row["operation"]
            app, verb, nouns = _operation_tokens(operation)
            if app not in named_apps:
                continue
            row_tokens = set(intent_tokens)
            imperative = verb == "create" and re.search(
                rf"\b{re.escape(app)}\s+(?:him|her|them)\b", instruction, re.I)
            if imperative:
                row_tokens.update({"user", "recipient", "receiver"})
            action = sorted(_ACTION_SURFACES.get(verb, frozenset({verb})) & row_tokens)
            # A named app used as a verb ("App her") is an action only in
            # that public imperative form, never from a bare app mention.
            if imperative:
                action.append(app)
            if not action:
                continue
            objects = sorted({term for term in (*nouns, *row["description_tokens"])
                              if term not in _INTENT_STOP and term not in _ACTION_SURFACES.get(verb, ())})
            matched = {term: sorted(_CONCEPT_SURFACES.get(term, frozenset({term})) & row_tokens)
                       for term in objects}
            matched = {term: values for term, values in matched.items() if values}
            if not matched:
                continue
            # A generic payment request does not prove a transfer effect.
            if verb == "create" and "transaction" in nouns:
                direct = bool({"send", "pay", "transfer"} & set(action))
                if not ({"money", "transaction"} & set(matched)) or not ("user" in matched or direct):
                    continue
                if "request" in intent_tokens and not direct:
                    continue
            candidates.append({"operation": operation, "app": app, "verb": verb,
                               "matched_action_terms": action,
                               "matched_object_terms": [
                                   {"documented_term": term, "instruction_terms": values}
                                   for term, values in matched.items()],
                               "public_access_mode": "documented_safe_terminal_slice",
                               "match_mode": "named_app_action_and_object_public_intent"})
        intent_added = sorted(candidates, key=lambda row: row["operation"])
        # A base lexical match and a public intent match describe one callable.
        intent_operations = {item["operation"] for item in intent_added}
        matches = [row for row in matches if row["operation"] not in intent_operations] + intent_added
    matches.sort(key=lambda item: item["operation"])
    operations = [item["operation"] for item in matches]
    descriptor = {"domain": domain, "instruction_token_sha256": digest(sorted(tokens)),
                  "relevant_public_apps": list(named_apps), "candidate_operations": operations}
    record = {"policy_id": POLICY_ID,
              "instruction_sha256": hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
              "public_tool_metadata_sha256": digest(public_tool_metadata),
              "callable_registry_sha256": str(callable_registry["registry_sha256"]),
              "alias_map_sha256": digest(aliases), "derived_public_descriptor": descriptor,
              # ``query_operations`` retains the maintained runner boundary;
              # the canonical name is explicit for provenance.
              "canonical_query_operations": operations, "query_operations": operations,
              "operation_evidence": matches,
              "terminal_relaxation": {"added_operations": [item["operation"] for item in noun_added]},
              "derivation_status": "specific" if operations else "empty_public_query",
              "derivation_reason": None if operations else "no public operation has both a supported action and semantic component"}
    if intents is not None:
        record["intent_registry_sha256"] = intents["intent_registry_sha256"]
        record["safe_terminal_operation_evidence"] = intent_added
        record["safe_terminal_operations"] = [item["operation"] for item in intent_added]
        record["relaxed_relevant_public_apps"] = [] if _PAST_CONTEXT.search(instruction) else list(named_apps)
    record["query_sha256"] = digest(record)
    return record


def validate_task_query(record: Mapping[str, Any], *, instruction: str, public_tool_metadata: Mapping[str, Any],
                        callable_registry: Mapping[str, Any]) -> None:
    descriptor = record.get("derived_public_descriptor")
    if not isinstance(descriptor, Mapping):
        raise ValueError("task query has no public descriptor")
    expected = derive_task_query(instruction, str(descriptor.get("domain") or ""),
                                 public_tool_metadata, callable_registry)
    if dict(record) != expected:
        raise ValueError("task-query provenance does not reproduce from public inputs")
