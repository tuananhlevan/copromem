"""Registry-driven semantic projection for CoProMem.

Only response-attested domain calls can enter induced procedures.  Discovery,
authentication, runtime context, and supervisor control remain provenance-only.
"""
from __future__ import annotations

from typing import Any, Mapping

from .graph import Graph, build_graph, digest, plan_task_batch, validate_plan

POLICY_ID = "copromem-semantic-projection"
ROLE_POLICY_ID = "copromem-registry-semantic-role"
_AUTH_VERBS = frozenset({"login", "logout", "signup", "verify_account", "reset_password", "send_verification_code", "send_password_reset_code"})
_NAMESPACE_ROLES = {"api_docs": "infrastructure_discovery", "supervisor": "supervisor_control"}


def classify_operation(meta: Mapping[str, Any]) -> str:
    """Classify from frozen public registry metadata, never task outcomes."""
    app = str(meta.get("app", ""))
    name = str(meta.get("function_name", "")).rsplit("__", 1)[-1]
    if app in _NAMESPACE_ROLES:
        return _NAMESPACE_ROLES[app]
    # Runtime context is a field classification, not an operation role: a
    # domain mutation which accepts an access token is still a domain action.
    if name in _AUTH_VERBS:
        return "authentication_runtime_context"
    return "domain_operation"


def _index(registry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = registry.get("operations", [])
    result = {str(row.get("operation")): row for row in rows if isinstance(row, Mapping)}
    if not result or len(result) != len(rows):
        raise ValueError("frozen public callable registry is malformed")
    return result


def project_graph(graph: Graph, registry: Mapping[str, Any]) -> tuple[Graph, dict[str, Any]]:
    """Project non-domain scaffolding while preserving domain occurrence order.

    Occurrence identity is semantic evidence: without a general public proof
    that removing it is safe, repeated calls remain distinct.  In particular,
    ``read -> write -> read`` stays three nodes even if the reads share an API
    name and public slots.
    """
    index = _index(registry)
    roles: dict[int, str] = {}
    excluded: list[dict[str, Any]] = []
    for position, node in enumerate(graph.nodes):
        operation = str(node["operation"]); meta = index.get(operation)
        if meta is None:
            raise ValueError("graph operation absent from frozen registry")
        role = classify_operation(meta); roles[position] = role
        if role != "domain_operation":
            excluded.append({"node_index": position, "operation": operation, "role": role, "reason": "non_domain_provenance_only"})
    domain = {position for position, role in roles.items() if role == "domain_operation"}
    mapping: dict[int, int] = {}; retained: list[dict[str, Any]] = []
    for new_index, position in enumerate(sorted(domain)):
        node = dict(graph.nodes[position]); node["index"] = new_index
        node["multiplicity"] = 1; node["evidence_node_indices"] = [position]
        node["evidence_node_sha256"] = [digest(graph.nodes[position])]
        retained.append(node); mapping[position] = new_index
    edges: list[dict[str, Any]] = []
    # Preserve direct domain order and induce a transitive domain order across
    # excluded infrastructure/auth/control nodes.
    positions = sorted(domain)
    for left, right in zip(positions, positions[1:]):
        edges.append({"kind": "order", "from": mapping[left], "to": mapping[right],
                      "induced_over_excluded": right - left > 1})
    for edge in graph.edges:
        if edge["kind"] not in {"redacted_dataflow", "declared_dependency"} or edge["from"] not in domain or edge["to"] not in domain:
            continue
        projected = {key: value for key, value in edge.items() if key not in {"from", "to"}}
        projected.update({"from": mapping[int(edge["from"])], "to": mapping[int(edge["to"])]})
        if projected not in edges: edges.append(projected)
    edges = sorted(edges, key=lambda item: (int(item["from"]), int(item["to"]), str(item["kind"]), digest(item)))
    body = {"registry_sha256": graph.registry_sha256, "nodes": tuple(retained), "edges": tuple(edges)}
    semantic = Graph(graph.registry_sha256, body["nodes"], body["edges"], digest(body))
    audit = {"policy_id": POLICY_ID, "role_policy_id": ROLE_POLICY_ID, "original_graph_sha256": graph.sha256,
             "semantic_projection_sha256": semantic.sha256, "excluded_nodes": excluded, "collapsed_repeats": [],
             "retained_domain_nodes": [{"operation": row["operation"], "multiplicity": row["multiplicity"],
                                         "evidence_node_sha256": row["evidence_node_sha256"]} for row in retained],
             "retained_domain_dataflow_edges": edges, "provenance_sha256": digest({"original": graph.sha256, "semantic": semantic.sha256, "excluded": excluded, "collapsed": [], "edges": edges})}
    return semantic, audit


def semantic_plan(success: list[Graph], failed: list[Graph], pre_state: Mapping[str, Any], audits: list[dict[str, Any]]) -> dict[str, Any]:
    plan = plan_task_batch(success, failed, pre_state)
    schema = dict(plan["schema"])
    schema.update({"policy_id": POLICY_ID, "semantic_projection_hashes": [audit["semantic_projection_sha256"] for audit in audits],
                   "semantic_provenance_hashes": [audit["provenance_sha256"] for audit in audits]})
    # A supported terminal may be a public read as well as a mutation.  The
    # base planner prefers writes; use the final semantic core operation
    # only when no write is present.
    if schema.get("terminal_effect") is None and schema.get("required_operations"):
        schema["terminal_effect"] = schema["required_operations"][-1]
    plan = {**plan, "schema": schema, "planner_id": POLICY_ID}
    plan["schema_id"] = f"schema_{digest(schema)[:16]}"
    plan["plan_sha256"] = digest({key: value for key, value in plan.items() if key != "plan_sha256"})
    return plan


def validate_semantic_plan(plan: Mapping[str, Any], registry: Mapping[str, Any]) -> dict[str, Any]:
    base = validate_plan(plan); index = _index(registry); schema = plan.get("schema", {})
    required = list(schema.get("required_operations", ())) if isinstance(schema, Mapping) else []
    terminal = schema.get("terminal_effect") if isinstance(schema, Mapping) else None
    roles = [classify_operation(index[operation]) if operation in index else "unknown" for operation in required]
    terminal_role = classify_operation(index[terminal]) if terminal in index else "unknown"
    reason = None
    if not required: reason = "no_domain_operation"
    elif any(role != "domain_operation" for role in roles): reason = "non_domain_core_step"
    elif terminal_role != "domain_operation": reason = "unsupported_terminal_effect"
    elif not base["passed"]: reason = str(base["reason"])
    return {"passed": reason is None, "reason": reason, "plan_sha256": plan.get("plan_sha256"),
            "required_operation_count": len(required), "terminal_role": terminal_role}


def build_semantic_graph(records: list[Mapping[str, Any]], registry: Mapping[str, Any]) -> tuple[Graph, dict[str, Any]]:
    return project_graph(build_graph(records, registry), registry)
