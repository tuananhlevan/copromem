"""Deterministic, value-redacted contrastive execution-graph memory.

This module has no model or benchmark dependency.  It accepts only dispatcher
evidence produced by ``public-execution-evidence`` plus aggregate success.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


POLICY_ID = "copromem-contrastive-graph"


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


@dataclass(frozen=True)
class Graph:
    registry_sha256: str
    nodes: tuple[dict[str, Any], ...]
    edges: tuple[dict[str, Any], ...]
    sha256: str


def _node(row: Mapping[str, Any], registry: Mapping[str, Any]) -> dict[str, Any]:
    signature = row.get("operation_signature")
    if not isinstance(signature, Mapping) or not row.get("schema_accepted") or not row.get("response_success"):
        raise ValueError("only response-attested schema-valid events may enter graphs")
    operation = signature.get("operation")
    meta = next((item for item in registry.get("operations", []) if item.get("operation") == operation), None)
    if not isinstance(meta, Mapping):
        raise ValueError("event operation is absent from frozen callable registry")
    # Invocation and response digests are retained only as equality witnesses.
    if signature.get("public_required") != list(meta.get("required_parameters", [])) or signature.get("output_slots") != list(meta.get("output_slots", [])):
        raise ValueError("event public slots differ from frozen registry")
    inputs = row.get("invocation_value_hashes", {})
    outputs = row.get("response_output_value_hashes", {})
    if not isinstance(inputs, Mapping) or not isinstance(outputs, Mapping):
        raise ValueError("evidence does not contain redacted dataflow hashes")
    return {"operation": operation, "application": signature.get("application"),
            "callable_name": signature.get("callable_name"), "effect_class": meta.get("access_mode"),
            "inputs": sorted((str(key), str(inputs[key])) for key in meta.get("required_parameters", []) if key in inputs),
            "outputs": sorted((str(key), str(outputs[key])) for key in meta.get("output_slots", []) if key in outputs),
            "public_required": list(signature.get("public_required", [])),
            "output_slots": list(signature.get("output_slots", [])),
            "index": int(row["monotonic_index"]), "response_class": str(row.get("response_error_class") or "success")}


def build_graph(records: Iterable[Mapping[str, Any]], registry: Mapping[str, Any]) -> Graph:
    """Build a replayable public graph; concrete values never leave telemetry."""
    registry_hash = str(registry.get("registry_sha256", ""))
    if not registry_hash:
        raise ValueError("frozen callable registry required")
    nodes = tuple(_node(row, registry) for row in sorted(records, key=lambda item: int(item["monotonic_index"])))
    edges: list[dict[str, Any]] = []
    for right in range(1, len(nodes)):
        edges.append({"kind": "order", "from": right - 1, "to": right})
    declared = {(row["from_operation"], row["to_operation"]) for row in registry.get("dependency_edges", [])}
    for left, producer in enumerate(nodes):
        outputs = dict(producer["outputs"])
        for right, consumer in enumerate(nodes[left + 1:], left + 1):
            for output_name, output_hash in outputs.items():
                for input_name, input_hash in dict(consumer["inputs"]).items():
                    if output_hash == input_hash:
                        edges.append({"kind": "redacted_dataflow", "from": left, "to": right,
                                      "producer_slot": output_name, "consumer_slot": input_name})
            if (producer["operation"], consumer["operation"]) in declared:
                edges.append({"kind": "declared_dependency", "from": left, "to": right})
    body = {"registry_sha256": registry_hash, "nodes": nodes, "edges": sorted(edges, key=lambda item: canonical(item))}
    return Graph(registry_hash, nodes, tuple(body["edges"]), digest(body))


def _core(graphs: list[Graph]) -> list[str]:
    """Greedy ordered scan of operation/access-mode identities."""
    sequences = [[(node["operation"], node["effect_class"]) for node in graph.nodes] for graph in graphs]
    first = sequences[0]
    result: list[tuple[str, str]] = []
    positions = [0] * len(sequences)
    for item in first:
        found: list[int] = []
        for sequence, position in zip(sequences, positions):
            try: found.append(sequence.index(item, position))
            except ValueError: break
        else:
            result.append(item); positions = [position + 1 for position in found]
    return [operation for operation, _ in result]


def plan_task_batch(success_graphs: Iterable[Graph], failed_graphs: Iterable[Graph], pre_state: Mapping[str, Any], *, trial_count: int | None = None) -> dict[str, Any]:
    """Keep the greedy ordered core of successful trials, as specified by the paper.

    Failed trials remain audit evidence; they cannot remove a successful core step.
    """
    success = list(success_graphs)
    failed = list(failed_graphs)
    count = len(success) + len(failed) if trial_count is None else trial_count
    if count < 1 or count != len(success) + len(failed):
        raise ValueError("trial count must equal the complete scored batch")
    registry_hash = success[0].registry_sha256 if success else ""
    if success and any(graph.registry_sha256 != registry_hash for graph in success + failed):
        raise ValueError("mixed registry evidence")
    core = _core(success) if success else []
    first = success[0] if success else None
    aligned = []
    cursor = 0
    for operation in core:
        position = next(i for i in range(cursor, len(first.nodes)) if first.nodes[i]["operation"] == operation)
        aligned.append(first.nodes[position])
        cursor = position + 1
    terminal_position = next((i for i in range(len(aligned) - 1, -1, -1)
                              if aligned[i]["effect_class"] == "write"), len(aligned) - 1)
    terminal = core[terminal_position] if core else None
    schema = {"policy_id": POLICY_ID, "registry_sha256": registry_hash,
              "required_operations": core, "terminal_effect": terminal,
              "typed_constraints": [{"operation": node["operation"], "required": node["public_required"],
                                     "outputs": node["output_slots"]} for node in aligned],
              "support": {"successes": len(success), "failures": len(failed), "trial_count": count,
                          "min_successes": 1 if count == 1 else 2},
              "graph_hashes": sorted(graph.sha256 for graph in success + failed)}
    candidate = {"schema": schema, "schema_id": f"schema_{digest(schema)[:16]}",
                 "pre_state_sha256": digest(pre_state), "min_successes": 1 if count == 1 else 2,
                 "planner_id": POLICY_ID}
    candidate["plan_sha256"] = digest(candidate)
    return candidate


def validate_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    schema = plan.get("schema", {})
    support = schema.get("support", {}) if isinstance(schema, Mapping) else {}
    required = schema.get("required_operations", []) if isinstance(schema, Mapping) else []
    terminal = schema.get("terminal_effect") if isinstance(schema, Mapping) else None
    passed = bool(len(required) and terminal in required and int(support.get("successes", 0)) >= int(plan.get("min_successes", 2)))
    return {"passed": passed, "reason": None if passed else "insufficient_contrastive_response_attested_evidence",
            "plan_sha256": plan.get("plan_sha256"), "required_operation_count": len(required)}
