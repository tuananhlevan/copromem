from __future__ import annotations

from copy import deepcopy

import pytest

from copromem import (
    CoProMem,
    admission_receipt,
    build_intent_registry,
    multi_schema_admission_bundle,
    schema_identity,
)
from copromem.retrieval import external_retrieval_policy, reproduce_retrieval
from copromem.spine import POLICY_ID as LEARNING_POLICY_ID


def _dataset() -> dict:
    operations = [
        {"operation": "tools.phone.search_contacts", "app": "phone", "http_method": "GET",
         "access_mode": "read", "required_parameters": ["query"], "output_slots": ["contact_id"]},
        {"operation": "tools.phone.send_text_message", "app": "phone", "http_method": "POST",
         "access_mode": "write", "required_parameters": ["body"], "output_slots": ["message_id"]},
        {"operation": "tools.venmo.create_transaction", "app": "venmo", "http_method": "POST",
         "access_mode": "write", "required_parameters": ["amount", "receiver"], "output_slots": ["transaction_id"]},
    ]
    return {
        "name": "public-safe_retrieval-smoke",
        "operations": operations,
        "app_descriptions": {"phone": "Public messaging application", "venmo": "Public payments application"},
        "public_operation_intents": build_intent_registry([
            {"operation": "tools.phone.send_text_message", "description": "Send a text message."},
            {"operation": "tools.venmo.create_transaction", "description": "Send money to a user."},
        ]),
    }


def _schema(registry_sha256: str, terminal: str, *, with_unproven_prior: bool = False) -> dict:
    required = (["tools.phone.search_contacts"] if with_unproven_prior else []) + [terminal]
    typed = []
    if with_unproven_prior:
        typed.append({"operation": "tools.phone.search_contacts", "occurrence_index": 1,
                      "required": ["query"], "outputs": ["contact_id"]})
    typed.append({"operation": terminal, "occurrence_index": len(typed) + 1,
                  "required": ["body"] if ".phone." in terminal else ["amount", "receiver"],
                  "outputs": ["message_id"] if ".phone." in terminal else ["transaction_id"]})
    return {
        "policy_id": LEARNING_POLICY_ID,
        "registry_sha256": registry_sha256,
        "required_operations": required,
        "terminal_effect": terminal,
        "terminal_occurrence_id": f"{terminal}#1",
        "typed_constraints": typed,
        "support": {"successes": 2, "min_successes": 2},
    }


def _admission(state: dict) -> dict:
    policy = external_retrieval_policy()["policy_sha256"]
    receipts = []
    for schema_id, schema in state["schemas"].items():
        receipts.append(admission_receipt(
            bank_sha256=f"source-{schema_id}", schema_ids=[schema_id],
            validation_task_id=f"heldout-{schema_id}", artifact_sha256="a" * 64,
            scorer_evidence_sha256="b" * 64, journal_sha256="c" * 64, passed=True,
            retrieval_policy_sha256=policy, schema_identity_sha256=schema_identity(schema),
        ))
    return multi_schema_admission_bundle(state, receipts, policy_sha256=policy)


def _memory(state: dict) -> CoProMem:
    return CoProMem(_dataset(), state=state, admission=_admission(state))


def test_safe_retrieval_unique_relevant_app_terminal_is_rendered_without_prerequisites() -> None:
    base = CoProMem(_dataset())
    state = {"schemas": {"phone": _schema(base.registry["registry_sha256"],
                                             "tools.phone.send_text_message", with_unproven_prior=True)}}
    memory = _memory(state)
    prepared = memory.prepare({"task_id": "phone-account", "domain": "demo",
                               "instruction": "Update my Phone account settings."})
    assert "tools.phone.send_text_message" in prepared["guidance"]
    assert "tools.phone.search_contacts" not in prepared["guidance"]
    assert prepared["provenance"]["selection_decision"] == "admitted_relevant_app_terminal_slice"
    assert reproduce_retrieval(memory.state, prepared["query"], memory.registry,
                               prepared["provenance"], memory.admission) == prepared["guidance"]


def test_safe_retrieval_rejects_cross_app_and_past_context() -> None:
    base = CoProMem(_dataset())
    state = {"schemas": {"phone": _schema(base.registry["registry_sha256"], "tools.phone.send_text_message")}}
    memory = _memory(state)
    cross_app = memory.prepare({"task_id": "venmo", "domain": "demo",
                                "instruction": "Send money on Venmo to Anita."})
    assert cross_app["guidance"] == ""
    assert cross_app["provenance"]["candidate_scores"][0]["rejection_reason"] == "terminal_app_not_publicly_relevant"
    past = memory.prepare({"task_id": "past", "domain": "demo",
                           "instruction": "I used Phone yesterday; update my account."})
    assert past["guidance"] == ""


def test_safe_retrieval_abstains_for_distinct_terminal_slices_without_id_or_support_tiebreaking() -> None:
    base = CoProMem(_dataset())
    state = {"schemas": {
        "z-phone": _schema(base.registry["registry_sha256"], "tools.phone.send_text_message"),
        "a-venmo": _schema(base.registry["registry_sha256"], "tools.venmo.create_transaction"),
    }}
    memory = _memory(state)
    prepared = memory.prepare({"task_id": "both", "domain": "demo",
                               "instruction": "Use Phone and Venmo for my account."})
    assert prepared["guidance"] == ""
    assert prepared["provenance"]["selection_decision"] == "semantic_tie_abstention"
    changed = deepcopy(state)
    changed["schemas"]["z-phone"]["support"]["successes"] = 99
    changed_memory = _memory(changed)
    assert changed_memory.prepare({"task_id": "both", "domain": "demo",
                                   "instruction": "Use Phone and Venmo for my account."})["guidance"] == ""


def test_safe_retrieval_tampered_admission_or_provenance_fails_closed() -> None:
    base = CoProMem(_dataset())
    state = {"schemas": {"phone": _schema(base.registry["registry_sha256"], "tools.phone.send_text_message")}}
    memory = _memory(state)
    task = {"task_id": "phone", "domain": "demo", "instruction": "Send a message on Phone."}
    prepared = memory.prepare(task)
    bad = deepcopy(prepared["provenance"])
    bad["guidance"] = "forged"
    with pytest.raises(ValueError, match="provenance mismatch"):
        reproduce_retrieval(memory.state, prepared["query"], memory.registry, bad, memory.admission)
    bad_bundle = deepcopy(memory.admission)
    bad_bundle["entries"][0]["schema_identity_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="bundle hash mismatch"):
        CoProMem(_dataset(), state=state, admission=bad_bundle).prepare(task)
