from __future__ import annotations

import io
import json
from pathlib import Path
import pytest

from copromem import (CoProMem, OpenRouterProvider, ScriptedProvider, Tool, ToolAgent,
                      ToolRegistry, admit_agent_report, build_intent_registry,
                      multi_schema_admission_bundle, run_agent)


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "toy"


def test_scripted_agent_runs_scored_batch_before_memory_update(tmp_path: Path) -> None:
    for name in ("dataset.json", "tasks.jsonl", "agent_run.json", "agent_responses.json", "agent_env.py"):
        (tmp_path / name).write_bytes((EXAMPLE / name).read_bytes())
    report = run_agent(tmp_path / "agent_run.json")
    result = report["results"][0]
    assert result["prepared"]["guidance"] == ""
    assert [trial["success"] for trial in result["trials"]] == [True, True]
    assert [len(trial["calls"]) for trial in result["trials"]] == [2, 2]
    assert result["update"]["marker"]["state"] == "committed"
    assert len(report["bank"]["schemas"]) == 1
    assert result["pre_state_sha256"] != result["post_state_sha256"]
    for trial in result["trials"]:
        assert trial["messages"][0]["role"] == "system"
        assert any(message["role"] == "tool" for message in trial["messages"])
    assert json.loads((tmp_path / "output" / "agent_result.json").read_text()) == report


def test_later_task_receives_learned_guidance(tmp_path: Path) -> None:
    for name in ("dataset.json", "tasks.jsonl", "agent_run.json", "agent_responses.json", "agent_env.py"):
        (tmp_path / name).write_bytes((EXAMPLE / name).read_bytes())
    first = json.loads((tmp_path / "tasks.jsonl").read_text())
    second = {**first, "task_id": "later-task"}
    (tmp_path / "tasks.jsonl").write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")
    responses = json.loads((tmp_path / "agent_responses.json").read_text())
    (tmp_path / "agent_responses.json").write_text(json.dumps(responses + responses))
    report = run_agent(tmp_path / "agent_run.json")
    first_result, later = report["results"]
    assert first_result["prepared"]["guidance"] == ""
    assert "tools.music.follow_artist" in later["prepared"]["guidance"]
    assert later["pre_state_sha256"] == first_result["post_state_sha256"]
    assert later["trials"][0]["guidance_sha256"] == later["prepared"]["provenance"]["guidance_sha256"]


def test_agent_respects_custom_episode_and_call_field_paths(tmp_path: Path) -> None:
    for name in ("dataset.json", "tasks.jsonl", "agent_run.json", "agent_responses.json", "agent_env.py"):
        (tmp_path / name).write_bytes((EXAMPLE / name).read_bytes())
    dataset_path = tmp_path / "dataset.json"
    dataset = json.loads(dataset_path.read_text())
    dataset["episode_fields"] = {"task_id": "identity.id", "success": "grade.passed", "calls": "trace.calls"}
    dataset["call_fields"] = {"operation": "api.name", "inputs": "api.args", "outputs": "api.result",
                              "schema_accepted": "checks.schema", "response_success": "checks.success"}
    dataset_path.write_text(json.dumps(dataset))
    report = run_agent(tmp_path / "agent_run.json")
    assert report["results"][0]["update"]["marker"]["state"] == "committed"
    assert len(report["bank"]["schemas"]) == 1


def test_invalid_arguments_do_not_execute_or_enter_learning_evidence() -> None:
    seen = []
    tool = Tool(name="find_item", operation="tools.shop.find_item", description="Find an item",
                parameters={"type": "object", "properties": {"name": {"type": "string"}},
                            "required": ["name"]}, output_slots=("item_id",),
                handler=lambda name: seen.append(name) or {"item_id": "item-1"})
    provider = ScriptedProvider([
        {"role": "assistant", "content": None, "tool_calls": [{"id": "bad", "type": "function",
            "function": {"name": "find_item", "arguments": '{"name": 5}'}}]},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "good", "type": "function",
            "function": {"name": "find_item", "arguments": '{"name": "book"}'}}]},
        {"role": "assistant", "content": "Found it."},
    ])
    result = ToolAgent(provider).run("Find a book", "Use prior guidance", ToolRegistry([tool]))
    assert seen == ["book"]
    assert [event["response_success"] for event in result["events"]] == [False, True]
    assert len(result["calls"]) == 1 and result["calls"][0]["schema_accepted"] is True
    assert "Use prior guidance" in provider.requests[0]["messages"][0]["content"]
    assert provider.requests[1]["messages"][-1]["role"] == "tool"
    assert provider.requests[1]["tools"] == provider.requests[0]["tools"]


def test_openrouter_adapter_sends_tool_call_contract_without_network() -> None:
    captured = {}

    def opener(request, *, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        response = {"id": "response-1", "model": "example/model", "usage": {"total_tokens": 5},
                    "choices": [{"message": {"role": "assistant", "content": None,
                     "tool_calls": [{"id": "call-1", "type": "function",
                                     "function": {"name": "lookup", "arguments": "{}"}}]}}]}
        return io.BytesIO(json.dumps(response).encode())

    provider = OpenRouterProvider("example/model", api_key="test-key", opener=opener)
    message = provider.complete([{"role": "user", "content": "Lookup"}],
                                [{"type": "function", "function": {"name": "lookup",
                                  "parameters": {"type": "object", "properties": {}}}}])
    request = captured["request"]
    body = json.loads(request.data)
    assert request.full_url == OpenRouterProvider.DEFAULT_URL
    assert request.get_header("Authorization") == "Bearer test-key"
    assert body["model"] == "example/model" and body["parallel_tool_calls"] is False
    assert body["tools"][0]["function"]["name"] == "lookup"
    assert message["tool_calls"][0]["id"] == "call-1"
    assert message["_provider"] == {"id": "response-1", "model": "example/model", "usage": {"total_tokens": 5}}


def test_agent_rejects_registry_drift_before_model_call(tmp_path: Path) -> None:
    for name in ("dataset.json", "tasks.jsonl", "agent_run.json", "agent_responses.json", "agent_env.py"):
        (tmp_path / name).write_bytes((EXAMPLE / name).read_bytes())
    dataset = json.loads((tmp_path / "dataset.json").read_text())
    dataset["operations"][0]["output_slots"] = ["different_slot"]
    (tmp_path / "dataset.json").write_text(json.dumps(dataset))
    with pytest.raises(ValueError, match="frozen public registry"):
        run_agent(tmp_path / "agent_run.json")


def test_agent_candidate_validation_to_admitted_terminal_bundle(tmp_path: Path) -> None:
    for name in ("dataset.json", "tasks.jsonl", "agent_run.json", "agent_responses.json",
                 "agent_env.py", "episodes.jsonl"):
        (tmp_path / name).write_bytes((EXAMPLE / name).read_bytes())
    dataset_path = tmp_path / "dataset.json"
    dataset = json.loads(dataset_path.read_text())
    dataset["public_operation_intents"] = build_intent_registry([{
        "operation": "tools.music.follow_artist", "description": "Follow an artist."}])
    dataset_path.write_text(json.dumps(dataset))
    memory = CoProMem(dataset)
    episodes = [json.loads(line) for line in (tmp_path / "episodes.jsonl").read_text().splitlines()]
    marker = memory.learn(episodes[0]["task_id"], episodes)["marker"]
    assert marker["state"] == "committed"
    schema_id = marker["winner_schema_id"]
    (tmp_path / "bank.json").write_text(json.dumps(memory.state))
    task = json.loads((tmp_path / "tasks.jsonl").read_text())
    config_path = tmp_path / "agent_run.json"
    config = json.loads(config_path.read_text())
    config.update({"initial_bank": "bank.json", "learn": False,
                   "candidate_validation": {"schema_id": schema_id, "task_id": task["task_id"]}})
    config_path.write_text(json.dumps(config))
    report = run_agent(config_path)
    prepared = report["results"][0]["prepared"]
    assert "tools.music.follow_artist" in prepared["guidance"]
    assert "tools.music.search_artist" not in prepared["guidance"]
    receipt = admit_agent_report(tmp_path / "output" / "agent_result.json",
                                 schema_id=schema_id, validation_task_id=task["task_id"])
    memory.admission = multi_schema_admission_bundle(memory.state, [receipt])
    assert memory.prepare(task)["guidance"] == prepared["guidance"]
