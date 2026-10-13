"""Zero-provider smoke pilot for CoProMem safe terminal-slice retrieval.

The example trains no model and calls no external benchmark.  It builds a
small admitted bank, retrieves a terminal-only public procedure, and prints
the reproducible provenance decision.
"""
from __future__ import annotations

from copromem import CoProMem, admission_receipt, build_intent_registry, multi_schema_admission_bundle, schema_identity
from copromem.retrieval import external_retrieval_policy
from copromem.spine import POLICY_ID


def main() -> None:
    dataset = {
        "name": "safe_retrieval-zero-provider-smoke",
        "operations": [{
            "operation": "tools.phone.send_text_message", "app": "phone", "http_method": "POST",
            "access_mode": "write", "required_parameters": ["body"], "output_slots": ["message_id"],
        }],
        "app_descriptions": {"phone": "Public messaging application"},
        "public_operation_intents": build_intent_registry([
            {"operation": "tools.phone.send_text_message", "description": "Send a text message."},
        ]),
    }
    seed = CoProMem(dataset)
    schema = {
        "policy_id": POLICY_ID, "registry_sha256": seed.registry["registry_sha256"],
        "required_operations": ["tools.phone.send_text_message"],
        "terminal_effect": "tools.phone.send_text_message",
        "terminal_occurrence_id": "tools.phone.send_text_message#1",
        "typed_constraints": [{"operation": "tools.phone.send_text_message", "occurrence_index": 1,
                               "required": ["body"], "outputs": ["message_id"]}],
        "support": {"successes": 2, "min_successes": 2},
    }
    state = {"schemas": {"phone-terminal": schema}}
    policy = external_retrieval_policy()["policy_sha256"]
    receipt = admission_receipt(
        bank_sha256="source-phone-bank", schema_ids=["phone-terminal"], validation_task_id="heldout-phone",
        artifact_sha256="a" * 64, scorer_evidence_sha256="b" * 64, journal_sha256="c" * 64,
        passed=True, retrieval_policy_sha256=policy, schema_identity_sha256=schema_identity(schema),
    )
    memory = CoProMem(dataset, state=state,
                     admission=multi_schema_admission_bundle(state, [receipt], policy_sha256=policy))
    prepared = memory.prepare({"task_id": "demo-phone", "domain": "demo",
                               "instruction": "Send a text message using Phone."})
    print(prepared["guidance"])
    print("decision:", prepared["provenance"]["selection_decision"])
    print("retrieval_sha256:", prepared["provenance"]["retrieval_sha256"])


if __name__ == "__main__":
    main()
