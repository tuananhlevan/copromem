from __future__ import annotations

from copy import deepcopy
import json

import pytest

from copromem import (CoProMem, admission_receipt, admit_agent_report, build_intent_registry,
                      build_intent_registry_from_openapi, multi_schema_admission_bundle,
                      external_retrieval_policy, schema_identity)
from copromem.admission import verify_admission
from copromem.graph import digest
from copromem.pipeline import _dataset_config
from copromem.retrieval import reproduce_retrieval
from copromem.spine import POLICY_ID as LEARNING_POLICY_ID


def _dataset(*operations: str, intents=None) -> dict:
    rows = [{"operation": operation, "app": operation.split(".")[1],
             "function_name": operation.split(".")[2], "access_mode": "write",
             "http_method": "POST", "required_parameters": [], "output_slots": ["result"]}
            for operation in operations]
    dataset = {"operations": rows,
               "app_descriptions": {operation.split(".")[1]: "public" for operation in operations}}
    if intents is not None:
        dataset["public_operation_intents"] = intents
    return dataset


def _schema(memory: CoProMem, operation: str, required=()) -> dict:
    return {"policy_id": LEARNING_POLICY_ID, "registry_sha256": memory.registry["registry_sha256"],
            "required_operations": [operation], "terminal_effect": operation,
            "terminal_occurrence_id": f"{operation}#1", "terminal_occurrence_ids": [f"{operation}#1"],
            "typed_constraints": [{"operation": operation, "occurrence_index": 1,
                                   "required": list(required), "outputs": ["result"]}],
            "support": {"successes": 2}}


def _bundle(memory: CoProMem, *schema_ids: str) -> dict:
    receipts = [admission_receipt(bank_sha256=digest(memory.state), schema_ids=[schema_id],
                                  validation_task_id=f"heldout-{schema_id}", artifact_sha256="artifact",
                                  scorer_evidence_sha256="score", journal_sha256="journal", passed=True,
                                  retrieval_policy_sha256=external_retrieval_policy()["policy_sha256"],
                                  schema_identity_sha256=schema_identity(memory.state["schemas"][schema_id]))
                for schema_id in schema_ids]
    return multi_schema_admission_bundle(memory.state, receipts)


def test_unique_named_app_noun_relaxes_terminal_but_ambiguous_noun_abstains() -> None:
    operation = "tools.music.follow_artist"
    memory = CoProMem(_dataset(operation))
    memory.state["schemas"]["one"] = _schema(memory, operation)
    task = {"task_id": "query", "domain": "music", "instruction": "In the music app, handle this artist."}
    prepared = memory.prepare(task)
    assert prepared["query"]["terminal_relaxation"]["added_operations"] == [operation]
    assert operation in prepared["guidance"]

    ambiguous = CoProMem(_dataset(operation, "tools.music.unfollow_artist"))
    ambiguous.state["schemas"]["one"] = _schema(ambiguous, operation)
    result = ambiguous.prepare(task)
    assert result["query"]["canonical_query_operations"] == []
    assert result["guidance"] == ""


def test_public_intent_requires_heldout_admission_and_reproduces() -> None:
    operation = "tools.venmo.create_transaction"
    intents = build_intent_registry([{"operation": operation, "description": "Send money to a user."}])
    memory = CoProMem(_dataset(operation, intents=intents))
    memory.state["schemas"]["one"] = _schema(memory, operation, ("amount", "receiver_email"))
    task = {"task_id": "query", "domain": "payments", "instruction": "Send $427 on Venmo to Anita."}
    with pytest.raises(ValueError, match="admission bundle"):
        memory.prepare(task)
    memory.admission = _bundle(memory, "one")
    result = memory.prepare(task)
    assert result["query"]["safe_terminal_operations"] == [operation]
    assert result["provenance"]["selection_decision"] == "admitted_exact_safe_terminal_slice"
    assert operation in result["guidance"]
    assert reproduce_retrieval(memory.state, result["query"], memory.registry,
                               result["provenance"], memory.admission) == result["guidance"]
    altered = deepcopy(result["provenance"])
    altered["guidance_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="provenance mismatch"):
        reproduce_retrieval(memory.state, result["query"], memory.registry, altered, memory.admission)
    memory.state["schemas"]["other"] = _schema(memory, operation)
    assert operation in memory.prepare(task)["guidance"]


def test_validation_is_isolated_and_admission_binds_schema_semantics() -> None:
    operation = "tools.venmo.create_transaction"
    intents = build_intent_registry([{"operation": operation, "description": "Send money to a user."}])
    memory = CoProMem(_dataset(operation, intents=intents))
    memory.state["schemas"]["one"] = _schema(memory, operation, ("amount", "receiver_email"))
    task = {"task_id": "heldout", "domain": "payments", "instruction": "Send $427 on Venmo to Anita."}
    result = memory.prepare_candidate_validation(task, schema_id="one", validation_task_id="heldout")
    assert result["provenance"]["retrieval_mode"] == "isolated_relevant_app_terminal_slice_candidate_validation"
    with pytest.raises(ValueError, match="task identity"):
        memory.prepare_candidate_validation(task, schema_id="one", validation_task_id="other")
    memory.admission = _bundle(memory, "one")
    memory.state["schemas"]["one"]["support"]["successes"] = 9
    assert operation in memory.prepare(task)["guidance"]
    memory.state["schemas"]["one"]["typed_constraints"][0]["required"] = ["different"]
    with pytest.raises(ValueError, match="semantic identity differs"):
        memory.prepare(task)


def test_guidance_contains_only_admitted_terminal_slice() -> None:
    terminal = "tools.venmo.create_transaction"
    lookup = "tools.venmo.lookup_account"
    intents = build_intent_registry([{"operation": terminal, "description": "Send money to a user."}])
    memory = CoProMem(_dataset(terminal, lookup, intents=intents))
    schema = _schema(memory, terminal, ("amount", "receiver_email"))
    schema["required_operations"] = [lookup, terminal]
    schema["typed_constraints"].insert(0, {"operation": lookup, "occurrence_index": 1,
                                           "required": ["private_token"], "outputs": ["receiver_email"]})
    schema["typed_constraints"][1]["occurrence_index"] = 2
    schema["attested_dataflow_edges"] = [{"from_occurrence_id": f"{lookup}#1",
                                           "to_occurrence_id": f"{terminal}#1",
                                           "producer_slot": "receiver_email", "consumer_slot": "receiver_email",
                                           "attestation": "redacted_value_equality_in_every_success"}]
    memory.state["schemas"]["unsafe"] = schema
    task = {"task_id": "query", "domain": "payments", "instruction": "Send $427 on Venmo to Anita."}
    memory.admission = _bundle(memory, "unsafe")
    result = memory.prepare(task)
    assert terminal in result["guidance"]
    assert lookup not in result["guidance"]
    assert "private_token" not in result["guidance"]
    assert "unproven prerequisites" in result["guidance"]

    memory.state["schemas"] = {"first": _schema(memory, terminal, ("amount", "receiver_email")),
                               "second": _schema(memory, terminal, ("amount", "receiver_email"))}
    memory.admission = _bundle(memory, "first", "second")
    result = memory.prepare(task)
    assert terminal in result["guidance"]
    assert result["provenance"]["selected_schema_ids"] == ["first", "second"]


def test_diverse_admitted_apps_and_distinct_terminal_tie() -> None:
    phone = "tools.phone.send_text_message"
    payment = "tools.venmo.create_transaction"
    intents = build_intent_registry([{"operation": phone, "description": "Send a text message."},
                                     {"operation": payment, "description": "Send money to a user."}])
    memory = CoProMem(_dataset(phone, payment, intents=intents))
    memory.state["schemas"]["phone"] = _schema(memory, phone, ("body",))
    memory.state["schemas"]["payment"] = _schema(memory, payment, ("amount",))
    memory.admission = _bundle(memory, "phone", "payment")
    task = {"task_id": "query", "domain": "communication",
            "instruction": "Send a text message using Phone."}
    result = memory.prepare(task)
    assert phone in result["guidance"] and payment not in result["guidance"]
    assert result["provenance"]["selected_schema_ids"] == ["phone"]

    task["instruction"] = "Send a message on Phone and transfer money with Venmo to Anita."
    result = memory.prepare(task)
    assert result["guidance"] == ""
    assert result["provenance"]["selection_decision"] == "semantic_tie_abstention"


def test_bundle_rejects_tampering_and_unadmitted_schema() -> None:
    operation = "tools.phone.send_text_message"
    intents = build_intent_registry([{"operation": operation, "description": "Send a text message."}])
    memory = CoProMem(_dataset(operation, intents=intents))
    memory.state["schemas"]["one"] = _schema(memory, operation)
    bundle = _bundle(memory, "one")
    memory.admission = bundle
    task = {"task_id": "query", "domain": "phone", "instruction": "Send a text message using Phone."}
    assert operation in memory.prepare(task)["guidance"]
    memory.admission = deepcopy(bundle)
    memory.admission["entries"][0]["schema_identity_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="bundle hash mismatch"):
        memory.prepare(task)
    memory.admission = bundle
    del memory.state["schemas"]["one"]
    with pytest.raises(ValueError, match="absent from live bank"):
        memory.prepare(task)


def test_relevant_app_terminal_relaxation_excludes_past_context() -> None:
    operation = "tools.venmo.create_transaction"
    intents = build_intent_registry([{"operation": operation, "description": "Send money to a user."}])
    memory = CoProMem(_dataset(operation, intents=intents))
    memory.state["schemas"]["one"] = _schema(memory, operation)
    memory.admission = _bundle(memory, "one")
    current = {"task_id": "current", "domain": "payments",
               "instruction": "Make a Venmo account for my parents."}
    result = memory.prepare(current)
    assert result["query"]["safe_terminal_operations"] == []
    assert result["provenance"]["selection_decision"] == "admitted_relevant_app_terminal_slice"
    assert operation in result["guidance"]
    historical = {"task_id": "later", "domain": "payments",
                  "instruction": "I paid on Venmo yesterday. Record it elsewhere."}
    result = memory.prepare(historical)
    assert result["guidance"] == ""


def test_openapi_descriptions_are_read_from_public_files(tmp_path) -> None:
    root = tmp_path / "openapi"
    root.mkdir()
    (root / "phone.json").write_text(json.dumps({"paths": {"/messages": {"post": {
        "operationId": "phone__send_text_message", "description": "Send a text message."}}}}))
    dataset = _dataset("tools.phone.send_text_message")
    intents = build_intent_registry_from_openapi(root, dataset["operations"])
    assert intents["operations"][0]["description"] == "Send a text message."
    assert intents["sources"][0]["name"] == "phone.json"
    CoProMem({**dataset, "public_operation_intents": intents})
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps({**dataset, "public_openapi_dir": "openapi"}))
    loaded = _dataset_config(tmp_path / "run.json", "dataset.json")
    assert loaded["public_operation_intents"] == intents
    (root / "phone.json").write_text(json.dumps({"paths": {"/messages": {"post": {
        "operationId": "phone__send_text_message", "description": "Send a short text message."}}}}))
    changed = _dataset_config(tmp_path / "run.json", "dataset.json")
    assert changed["public_operation_intents"]["intent_registry_sha256"] != intents["intent_registry_sha256"]


def test_intent_registry_rejects_description_tampering() -> None:
    operation = "tools.venmo.create_transaction"
    intents = build_intent_registry([{"operation": operation, "description": "Send money to a user."}])
    intents["operations"][0]["description"] = "Forged"
    with pytest.raises(ValueError, match="identity mismatch"):
        CoProMem(_dataset(operation, intents=intents))


def test_scored_candidate_artifact_creates_bank_bound_receipt(tmp_path) -> None:
    operation = "tools.venmo.create_transaction"
    intents = build_intent_registry([{"operation": operation, "description": "Send money to a user."}])
    memory = CoProMem(_dataset(operation, intents=intents))
    memory.state["schemas"]["one"] = _schema(memory, operation, ("amount", "receiver_email"))
    task = {"task_id": "heldout", "domain": "payments", "instruction": "Send $427 on Venmo to Anita."}
    prepared = memory.prepare_candidate_validation(task, schema_id="one", validation_task_id="heldout")
    bank_hash = digest(memory.state)
    messages = [{"role": "system", "content": "Retrieved procedure:\n" + prepared["guidance"]},
                {"role": "user", "content": task["instruction"]}]
    artifact = {"bank": memory.state, "registry": memory.registry,
                "results": [{"task_id": "heldout", "prepared": prepared,
                "pre_state_sha256": bank_hash, "post_state_sha256": bank_hash,
                "trials": [{"trial": 0, "success": True, "messages": messages,
                            "initial_prompt_sha256": digest(messages),
                            "guidance_sha256": digest(prepared["guidance"])}]}]}
    path = tmp_path / "validation.json"
    path.write_text(json.dumps(artifact))
    receipt = admit_agent_report(path, schema_id="one", validation_task_id="heldout")
    verify_admission(receipt, bank_sha256=bank_hash)
    memory.admission = multi_schema_admission_bundle(memory.state, [receipt])
    assert operation in memory.prepare(task)["guidance"]
    artifact["results"][0]["trials"][0]["success"] = False
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="did not pass"):
        admit_agent_report(path, schema_id="one", validation_task_id="heldout")
    artifact["results"][0]["trials"][0]["success"] = True
    artifact["results"][0]["trials"][0]["messages"][0]["content"] = "No guidance"
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="model-visible prompt"):
        admit_agent_report(path, schema_id="one", validation_task_id="heldout")
