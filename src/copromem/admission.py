"""Held-out receipts and semantic identities for admitted CoProMem schemas."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

from .graph import digest


def admission_receipt(*, bank_sha256: str, schema_ids: Sequence[str], validation_task_id: str,
                      artifact_sha256: str, scorer_evidence_sha256: str, journal_sha256: str,
                      passed: bool, retrieval_policy_sha256: str | None = None,
                      schema_identity_sha256: str | None = None) -> dict[str, Any]:
    ids = sorted(set(schema_ids))
    if (not bank_sha256 or not ids or len(ids) != len(schema_ids)
            or any(not isinstance(item, str) or not item for item in ids)
            or not all(isinstance(item, str) and item for item in
                       (validation_task_id, artifact_sha256, scorer_evidence_sha256, journal_sha256))
            or type(passed) is not bool):
        raise ValueError("external admission evidence is incomplete")
    row = {"bank_sha256": bank_sha256, "schema_ids": ids,
           "validation_task_id": validation_task_id, "artifact_sha256": artifact_sha256,
           "scorer_evidence_sha256": scorer_evidence_sha256,
           "journal_sha256": journal_sha256, "passed": passed}
    if retrieval_policy_sha256 is not None:
        if not isinstance(retrieval_policy_sha256, str) or len(retrieval_policy_sha256) != 64:
            raise ValueError("external admission retrieval policy identity is invalid")
        row["admission_kind"] = "retrieval_execution_contract"
        row["retrieval_policy_sha256"] = retrieval_policy_sha256
    if schema_identity_sha256 is not None:
        if len(ids) != 1 or not isinstance(schema_identity_sha256, str) or len(schema_identity_sha256) != 64:
            raise ValueError("external admission schema identity is invalid")
        row["schema_identity_sha256"] = schema_identity_sha256
    return {**row, "receipt_sha256": digest(row)}


def verify_admission(record: Mapping[str, Any], *, bank_sha256: str) -> None:
    row = dict(record)
    observed = row.pop("receipt_sha256", None)
    if observed != digest(row):
        raise ValueError("external admission receipt hash mismatch")
    if row.get("bank_sha256") != bank_sha256:
        raise ValueError("external admission receipt bank mismatch")
    if row.get("passed") is not True or not isinstance(row.get("schema_ids"), list) or not row["schema_ids"]:
        raise ValueError("schema is not externally admitted")
    if row["schema_ids"] != sorted(set(row["schema_ids"])):
        raise ValueError("external admission schema order is invalid")
    if not all(isinstance(row.get(key), str) and row[key] for key in
               ("validation_task_id", "artifact_sha256", "scorer_evidence_sha256", "journal_sha256")):
        raise ValueError("external admission evidence is incomplete")
    if "retrieval_policy_sha256" in row and (
            row.get("admission_kind") != "retrieval_execution_contract"
            or not isinstance(row["retrieval_policy_sha256"], str)
            or len(row["retrieval_policy_sha256"]) != 64):
        raise ValueError("external admission retrieval policy identity is invalid")
    if "schema_identity_sha256" in row and (
            len(row["schema_ids"]) != 1 or not isinstance(row["schema_identity_sha256"], str)
            or len(row["schema_identity_sha256"]) != 64):
        raise ValueError("external admission schema identity is invalid")


def schema_identity(schema: Mapping[str, Any]) -> str:
    """Bind immutable retrieval semantics while allowing support to grow."""
    return digest({"policy_id": schema.get("policy_id"),
                   "registry_sha256": schema.get("registry_sha256"),
                   "required_operations": list(schema.get("required_operations", ())),
                   "terminal_effect": schema.get("terminal_effect"),
                   "terminal_occurrence_id": schema.get("terminal_occurrence_id"),
                   "typed_constraints": list(schema.get("typed_constraints", ()))})


def multi_schema_admission_bundle(state: Mapping[str, Any],
                                  receipts: Sequence[Mapping[str, Any]], *,
                                  policy_sha256: str | None = None) -> dict[str, Any]:
    """Bundle independently validated schemas from one or more source banks."""
    if policy_sha256 is None:
        from .retrieval import external_retrieval_policy
        policy_sha256 = external_retrieval_policy()["policy_sha256"]
    schemas = state.get("schemas", {})
    entries = []
    for receipt in receipts:
        verify_admission(receipt, bank_sha256=str(receipt.get("bank_sha256") or ""))
        if receipt.get("retrieval_policy_sha256") != policy_sha256:
            raise ValueError("external admission retrieval policy differs")
        ids = receipt["schema_ids"]
        if len(ids) != 1 or ids[0] not in schemas:
            raise ValueError("admitted schema is absent or receipt covers multiple schemas")
        identity = schema_identity(schemas[ids[0]])
        if receipt.get("schema_identity_sha256") != identity:
            raise ValueError("external admission schema semantic identity differs")
        entries.append({"schema_id": ids[0], "schema_identity_sha256": identity,
                        "external_admission_receipt": dict(receipt)})
    entries.sort(key=lambda row: row["schema_id"])
    if not entries or len({row["schema_id"] for row in entries}) != len(entries):
        raise ValueError("multi-schema admission ids are empty or duplicated")
    value = {"retrieval_policy_sha256": policy_sha256, "entries": entries}
    return {**value, "bundle_sha256": digest(value)}


def verify_multi_schema_admission(bundle: Mapping[str, Any], *, policy_sha256: str,
                                  schemas: Mapping[str, Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    row = dict(bundle)
    observed = row.pop("bundle_sha256", None)
    if observed != digest(row):
        raise ValueError("multi-schema admission bundle hash mismatch")
    if row.get("retrieval_policy_sha256") != policy_sha256:
        raise ValueError("multi-schema admission policy differs")
    entries = row.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("multi-schema admission has no entries")
    admitted = {}
    previous = ""
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("multi-schema admission entry is malformed")
        schema_id = entry.get("schema_id")
        receipt = entry.get("external_admission_receipt")
        if not isinstance(schema_id, str) or not schema_id or schema_id <= previous:
            raise ValueError("multi-schema admission ids are unordered or duplicated")
        previous = schema_id
        if not isinstance(receipt, Mapping) or receipt.get("schema_ids") != [schema_id]:
            raise ValueError("multi-schema admission receipt does not bind exactly one schema")
        verify_admission(receipt, bank_sha256=str(receipt.get("bank_sha256") or ""))
        if receipt.get("retrieval_policy_sha256") != policy_sha256:
            raise ValueError("multi-schema admission receipt policy differs")
        schema = schemas.get(schema_id)
        if not isinstance(schema, Mapping):
            raise ValueError("admitted schema is absent from live bank")
        if (entry.get("schema_identity_sha256") != schema_identity(schema)
                or receipt.get("schema_identity_sha256") != entry.get("schema_identity_sha256")):
            raise ValueError("admitted schema semantic identity differs")
        admitted[schema_id] = schema
    return admitted


def admit_agent_report(artifact_path: str | Path, *, schema_id: str,
                       validation_task_id: str, journal_path: str | Path | None = None) -> dict[str, Any]:
    """Admit a successful isolated candidate run from the generic agent CLI."""
    artifact_path = Path(artifact_path)
    raw = artifact_path.read_bytes()
    report = json.loads(raw)
    results = report.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise ValueError("validation artifact must contain one task")
    result = results[0]
    provenance = result.get("prepared", {}).get("provenance", {})
    if (result.get("task_id") != validation_task_id
            or provenance.get("retrieval_mode") != "isolated_relevant_app_terminal_slice_candidate_validation"
            or provenance.get("candidate_schema_id") != schema_id
            or provenance.get("validation_task_id") != validation_task_id):
        raise ValueError("validation artifact task or candidate identity mismatch")
    bank = report.get("bank")
    bank_sha256 = digest(bank)
    if result.get("pre_state_sha256") != bank_sha256 or result.get("post_state_sha256") != bank_sha256:
        raise ValueError("validation artifact changed its frozen bank")
    from .retrieval import external_retrieval_policy, reproduce_retrieval
    policy_sha256 = external_retrieval_policy()["policy_sha256"]
    if provenance.get("policy_sha256") != policy_sha256:
        raise ValueError("validation artifact retrieval policy differs")
    guidance = reproduce_retrieval(bank, result["prepared"]["query"], report["registry"], provenance)
    if guidance != result["prepared"].get("guidance"):
        raise ValueError("validation artifact guidance does not reproduce")
    trials = result.get("trials")
    if not isinstance(trials, list) or not trials or any(row.get("success") is not True for row in trials):
        raise ValueError("held-out validation did not pass")
    for trial in trials:
        messages = trial.get("messages")
        if (not isinstance(messages, list) or len(messages) < 2
                or messages[0].get("role") != "system" or messages[1].get("role") != "user"
                or guidance not in str(messages[0].get("content") or "")
                or trial.get("initial_prompt_sha256") != digest(messages[:2])
                or trial.get("guidance_sha256") != digest(guidance)):
            raise ValueError("validation guidance was not bound to the model-visible prompt")
    scored = [{"trial": row["trial"], "success": row["success"]} for row in trials]
    journal_raw = Path(journal_path).read_bytes() if journal_path is not None else raw
    schema = bank.get("schemas", {}).get(schema_id)
    if not isinstance(schema, Mapping):
        raise ValueError("validation artifact candidate schema is absent from its bank")
    return admission_receipt(
        bank_sha256=bank_sha256, schema_ids=[schema_id], validation_task_id=validation_task_id,
        artifact_sha256=hashlib.sha256(raw).hexdigest(),
        scorer_evidence_sha256=digest(scored), journal_sha256=hashlib.sha256(journal_raw).hexdigest(),
        passed=True, retrieval_policy_sha256=policy_sha256,
        schema_identity_sha256=schema_identity(schema))
