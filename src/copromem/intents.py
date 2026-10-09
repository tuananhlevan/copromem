"""Content-addressed descriptions of public callable intent."""
from __future__ import annotations

import re
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .graph import digest


_TOKEN = re.compile(r"[a-z][a-z0-9_]{1,}")


def description_tokens(description: str) -> list[str]:
    return sorted(set(_TOKEN.findall(description.lower())))


def build_intent_registry(operations: Sequence[Mapping[str, Any]], *,
                          sources: Sequence[Mapping[str, str]] = ()) -> dict[str, Any]:
    """Freeze public descriptions before any task is dispatched."""
    rows = []
    for item in operations:
        operation = item.get("operation")
        description = item.get("description")
        if not isinstance(operation, str) or not operation or not isinstance(description, str) or not description.strip():
            raise ValueError("intent needs a public operation and description")
        rows.append({"operation": operation, "description": description,
                     "description_tokens": description_tokens(description)})
    rows.sort(key=lambda item: item["operation"])
    if not rows or len({row["operation"] for row in rows}) != len(rows):
        raise ValueError("intent operations must be nonempty and unique")
    record = {"operations": rows, **({"sources": list(sources)} if sources else {})}
    return {**record, "intent_registry_sha256": digest(record)}


def build_intent_registry_from_openapi(directory: str | Path,
                                       operations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Read public OpenAPI descriptions using the dataset's declared operations."""
    root = Path(directory)
    files = sorted(root.glob("*.json"))
    if not files:
        raise ValueError("public OpenAPI directory has no JSON files")
    by_app: dict[str, list[Mapping[str, Any]]] = {}
    for row in operations:
        by_app.setdefault(str(row.get("app") or row["operation"].split(".", 2)[1]), []).append(row)
    descriptions = []
    sources = []
    for path in files:
        raw = path.read_bytes()
        sources.append({"name": path.name, "sha256": hashlib.sha256(raw).hexdigest()})
        document = json.loads(raw)
        for methods in document.get("paths", {}).values():
            if not isinstance(methods, Mapping):
                continue
            for method, item in methods.items():
                if method.lower() not in {"get", "post", "put", "patch", "delete"} or not isinstance(item, Mapping):
                    continue
                operation_id = item.get("operationId")
                if not isinstance(operation_id, str):
                    continue
                action = operation_id.split("__")[-1].split(".")[-1]
                matched = [row for row in by_app.get(path.stem, ())
                           if (row["operation"] == operation_id
                               or row.get("function_name", row["operation"].split(".", 2)[2]) == action)]
                if len(matched) != 1:
                    raise ValueError(f"OpenAPI operation {operation_id} does not uniquely match the frozen registry")
                declared_method = matched[0].get("http_method")
                if declared_method and str(declared_method).upper() != method.upper():
                    raise ValueError(f"OpenAPI method for {operation_id} differs from the frozen registry")
                description = item.get("description") or item.get("summary")
                if not isinstance(description, str) or not description.strip():
                    raise ValueError(f"OpenAPI operation {operation_id} lacks a public description")
                descriptions.append({"operation": matched[0]["operation"], "description": description.strip()})
    return build_intent_registry(descriptions, sources=sources)


def verify_intent_registry(record: Mapping[str, Any], registry: Mapping[str, Any]) -> None:
    if not isinstance(record, Mapping):
        raise ValueError("public intent registry is malformed")
    rows = record.get("operations")
    if not isinstance(rows, list) or not rows or record.get("intent_registry_sha256") != digest(
            {key: value for key, value in record.items() if key != "intent_registry_sha256"}):
        raise ValueError("public intent registry identity mismatch")
    known = {row["operation"] for row in registry["operations"]}
    names = []
    for row in rows:
        if not isinstance(row, Mapping) or row.get("operation") not in known:
            raise ValueError("public intent operation is not in the frozen registry")
        if not isinstance(row.get("description"), str) or not row["description"].strip():
            raise ValueError("public intent description is malformed")
        if row.get("description_tokens") != description_tokens(row["description"]):
            raise ValueError("public intent tokens do not reproduce")
        names.append(row["operation"])
    if names != sorted(set(names)):
        raise ValueError("public intent operations are not unique and sorted")
